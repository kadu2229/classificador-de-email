# classificador-de-email
projeto de case prático para uma vaga que quero muito atuar

# AutoU - Classificador de Emails

Um projeto demo de classificação de emails usando FastAPI e aprendizado de máquina leve. Permite processar mensagens de texto, PDFs ou imagens e retorna uma classificação do email (Produtivo / Improdutivo), subtipo e uma resposta automática simulada.

---

## 🔹 Funcionalidades

- Processamento de **texto**, **PDFs** e **imagens**.
- Classificação de emails em **Produtivo** ou **Improdutivo**.
- Identificação de **subtipos**: status, anexo, suporte, agradecimento, felicitações.
- Geração de **resposta automática** baseada no conteúdo do email.
- Validação mínima de entrada: pelo menos 2 palavras.
- Interface web simples e responsiva.
- Efeito visual de fundo inspirado em lanternas de óleo (70s).

---

## ⚡ Tecnologias

- **Python 3.12**
- **FastAPI** (web framework)
- **Uvicorn** (ASGI server)
- **NLTK** e **Scikit-learn** (processamento de linguagem e classificação)
- **Pillow / pytesseract** (OCR em imagens)
- **pypdf / pdf2image** (leitura de PDFs)
- **HTML / CSS / JavaScript** (front-end)

---

## 🚀 Rodando localmente

1. Clone o repositório:

```bash
git clone <(https://github.com/kadu2229/classificador-de-email)>
cd <PASTA_DO_PROJETO>/backend


