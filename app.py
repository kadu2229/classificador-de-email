from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from typing import Tuple
import io
import re

# --- NLP / ML (leve, offline) ---
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# --- PDF / OCR ---
from pypdf import PdfReader
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes

app = FastAPI(title="AutoU - Classificador de Emails (Demo)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção, restrinja
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir o front simples
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

# --- NLTK setup ---
def _ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("stemmers/rslp")
    except LookupError:
        nltk.download("rslp")

_ensure_nltk()
PT_STOPWORDS = set(stopwords.words("portuguese"))
STEMMER = RSLPStemmer()

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[\d\W_]+", " ", text, flags=re.UNICODE)
    tokens = [w for w in text.split() if w not in PT_STOPWORDS and len(w) > 2]
    stems = [STEMMER.stem(w) for w in tokens]
    return " ".join(stems)

# --- Dataset sintético ---
PRODUTIVO = [
    "poderiam informar o status do meu protocolo 123456?",
    "segue dúvida sobre o acesso ao sistema, não estou conseguindo logar",
    "preciso de atualização do caso em aberto",
    "estou com erro na tela de pagamento",
    "não consigo recuperar a senha",
    "em anexo envio o comprovante para análise",
    "solicito suporte técnico",
    "poderiam retornar com a posição do atendimento?",
    "tenho uma dúvida sobre cadastro",
    "o arquivo anexado está correto?"
]
IMPRODUTIVO = [
    "feliz natal e um ótimo ano novo",
    "obrigado pela atenção",
    "parabéns pelo excelente trabalho",
    "bom dia, tenham uma ótima semana",
    "apenas agradecendo a ajuda",
    "boas festas a todos",
    "muito obrigado",
    "valeu demais",
    "mensagem sem necessidade de ação"
]

X_text = PRODUTIVO + IMPRODUTIVO
y = ["Produtivo"] * len(PRODUTIVO) + ["Improdutivo"] * len(IMPRODUTIVO)

VECT = TfidfVectorizer(preprocessor=preprocess)
X = VECT.fit_transform(X_text)

CLS = LogisticRegression(max_iter=1000)
CLS.fit(X, y)

KEYWORDS_IMPRODUTIVO = ["obrigado", "parabéns", "bom dia", "boa tarde", "boa noite", "feliz", "ótimo", "valeu"]

def classify_email_safe(text: str) -> Tuple[str, float]:
    lower = text.lower()
    for kw in KEYWORDS_IMPRODUTIVO:
        if kw in lower:
            return "Improdutivo", 0.99
    if not text or not text.strip():
        return "Improdutivo", 0.0
    Xq = VECT.transform([text])
    probs = CLS.predict_proba(Xq)[0]
    labels = CLS.classes_
    idx = int(np.argmax(probs))
    categoria = labels[idx]
    confianca = float(probs[idx])
    return categoria, confianca

SUBTYPES = {
    "status": r"\b(status|andament|protocolo|atualiza(ç|c)ao|posi(c|ç)ao)\b",
    "anexo": r"\b(anex|arquivo|documento|segue em anexo|em anexo)\b",
    "suporte": r"\b(erro|falha|bug|nao consigo|não consigo|indisponivel|indisponível|bloqueio|senha|acesso)\b",
    "agradecimento": r"\b(obrigad|valeu|agradec)\b",
    "felicitacoes": r"\b(feliz natal|boas festas|paraben|parabéns|bom dia|boa tarde|boa noite)\b"
}

def detect_subtype(text: str, categoria: str) -> str:
    t = text.lower()
    for name, pattern in SUBTYPES.items():
        if re.search(pattern, t):
            if name in ("agradecimento", "felicitacoes"):
                return name
            return name
    return "duvida" if categoria == "Produtivo" else "outros"

def extract_protocol(text: str) -> str | None:
    m = re.search(r"\b(\d{6,})\b", text)
    return m.group(1) if m else None

def generate_reply(text: str, categoria: str, subtipo: str) -> str:
    proto = extract_protocol(text)
    if categoria == "Produtivo":
        if subtipo == "status":
            if proto:
                return (f"Olá! Recebemos sua solicitação de status do protocolo {proto}. "
                        "Estamos verificando e retornamos em breve.")
            return ("Olá! Recebemos sua solicitação de status. "
                    "Se tiver um número de protocolo, por favor informe na resposta.")
        if subtipo == "anexo":
            return "Olá! Confirmamos o recebimento do arquivo/anexo. Retornaremos em breve."
        if subtipo == "suporte":
            return ("Olá! Sentimos pelo transtorno. Poderia informar passo a passo ou print da tela? "
                    "Nossa equipe já está acompanhando.")
        return "Olá! Encaminhamos sua dúvida para o time responsável e retornaremos com orientação."
    else:
        if subtipo == "agradecimento":
            return "Nós que agradecemos! Ficamos à disposição."
        if subtipo == "felicitacoes":
            return "Agradecemos os votos! Desejamos o mesmo para você. 😊"
        return "Obrigado pela mensagem! Seguimos à disposição."

# --- Função de leitura com OCR ---
def read_text_from_upload(file: UploadFile) -> str:
    if file.content_type.startswith("image/"):
        img = Image.open(file.file)
        return pytesseract.image_to_string(img, lang='por')

    elif file.content_type in ("application/pdf",):
        data = file.file.read()
        reader = PdfReader(io.BytesIO(data))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        # OCR em páginas sem texto
        if not pages_text:
            images = convert_from_bytes(data)
            for img in images:
                pages_text.append(pytesseract.image_to_string(img, lang='por'))
        return "\n".join(pages_text).strip()

    elif file.content_type in ("text/plain", "application/octet-stream"):
        raw = file.file.read()
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return raw.decode("latin-1", errors="ignore")

    else:
        raise HTTPException(status_code=400, detail="Formato não suportado. Use .txt, .pdf ou imagem")

# --- Rotas ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process")
async def process_email(email_text: str = Form(...)):
    categoria, conf = classify_email_safe(email_text)
    subtipo = detect_subtype(email_text, categoria)
    resposta = generate_reply(email_text, categoria, subtipo)
    return {
        "categoria": categoria,
        "confianca": round(conf, 3),
        "subtipo": subtipo,
        "resposta": resposta
    }

@app.post("/upload")
async def upload_email(file: UploadFile = File(...)):
    text = read_text_from_upload(file)
    if not text:
        raise HTTPException(status_code=400, detail="Não foi possível extrair texto do arquivo.")
    categoria, conf = classify_email_safe(text)
    subtipo = detect_subtype(text, categoria)
    resposta = generate_reply(text, categoria, subtipo)
    return {
        "categoria": categoria,
        "confianca": round(conf, 3),
        "subtipo": subtipo,
        "resposta": resposta,
        "chars_lidos": len(text)
    }
