# 🏥 MedCareAI

An AI-powered medical decision-support platform that helps users understand their symptoms and get reliable health information.

---

## 🎯 What Does This Project Do?

MedCareAI combines **3 AI systems** to help users:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MedCareAI Flow                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   👤 User: "I have headache and nausea"                             │
│                     │                                               │
│                     ▼                                               │
│   ┌─────────────────────────────────────┐                          │
│   │  Phase 1: ML Prediction             │                          │
│   │  (Random Forest + XGBoost)          │                          │
│   │  → "Prediction: Migraine (85%)"     │                          │
│   └─────────────────────────────────────┘                          │
│                     │                                               │
│                     ▼                                               │
│   ┌─────────────────────────────────────┐                          │
│   │  Phase 2: Chat Agent (Mistral AI)   │                          │
│   │  Asks follow-up questions           │                          │
│   │  → Extracts symptoms from chat      │                          │
│   └─────────────────────────────────────┘                          │
│                     │                                               │
│                     ▼                                               │
│   ┌─────────────────────────────────────┐                          │
│   │  Phase 3: RAG (ChromaDB + Mistral)  │                          │
│   │  → "Migraine is a neurological      │                          │
│   │     condition affecting 12% of      │                          │
│   │     adults... [Source: WHO]"        │                          │
│   └─────────────────────────────────────┘                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The 3 Phases Explained Simply:

| Phase | What It Does | Technology |
|-------|--------------|------------|
| **Phase 1: ML** | Predicts disease from symptoms | RandomForest (721 diseases, 377 symptoms) |
| **Phase 2: Chat** | Has a conversation to collect symptoms | Mistral AI (LLM) |
| **Phase 3: RAG** | Explains diseases with sources | ChromaDB + Mistral |

---

## 📁 Project Structure

```
medcareai/
├── backend/                    # Python FastAPI server
│   ├── app/                    # Web application
│   │   ├── api/v1/             # API endpoints
│   │   │   ├── predict.py      # /api/v1/predict - Disease prediction
│   │   │   ├── chat.py         # /api/v1/chat - Conversational AI
│   │   │   └── explain.py      # /api/v1/explain - RAG explanations
│   │   ├── services/           # Business logic
│   │   └── schemas/            # Data validation
│   ├── ml/                     # Machine Learning
│   │   ├── artifacts/          # Trained model + data (NOT in git)
│   │   ├── predictor.py        # Prediction logic
│   │   └── train.py            # Training script
│   ├── rag/                    # RAG Pipeline
│   │   ├── data/documents/     # Medical documents (txt/pdf)
│   │   ├── chunker.py          # Splits documents into chunks
│   │   ├── embedder.py         # Converts text to vectors
│   │   ├── vector_store.py     # ChromaDB database
│   │   └── rag_service.py      # RAG + LLM generation
│   ├── scripts/                # Utility scripts
│   │   └── rebuild_vectordb.py # Rebuild ChromaDB
│   ├── tests/                  # Unit tests
│   └── requirements.txt        # Python dependencies
├── frontend/                   # React web app (Vite + Tailwind)
│   ├── src/
│   │   ├── api/                # axios clients (predict / chat / explain)
│   │   ├── components/         # Layout, Spinner, etc.
│   │   ├── pages/              # Home, Predict, Chat, Explain
│   │   ├── App.jsx             # Routes
│   │   └── index.css           # Tailwind theme (black + beige)
│   ├── tailwind.config.js
│   └── package.json
├── .env.example                # Environment template
└── docker-compose.yml          # Docker setup
```

---

## 🚀 Quick Start (Step by Step)

### Prerequisites

You need these installed on your computer:

- **Python 3.11+** ([Download](https://www.python.org/downloads/))
- **Node.js 18+ and npm** ([Download](https://nodejs.org/)) — needed for the frontend
- **Git** ([Download](https://git-scm.com/downloads))
- **A Mistral API Key** ([Get free key](https://console.mistral.ai/))

### Step 1: Clone the Project

**On Linux/Mac:**
```bash
git clone https://github.com/yourusername/medcareai.git
cd medcareai
```

**On Windows (PowerShell):**
```powershell
git clone https://github.com/yourusername/medcareai.git
cd medcareai
```

### Step 2: Download the Dataset

The ML model needs training data. Download it from Kaggle:

1. Go to: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
2. Download `disease_symptom_dataset.csv`
3. Rename it to `diseases_symptoms.csv`
4. Put it in: `backend/ml/artifacts/diseases_symptoms.csv`

### Step 3: Set Up the Backend

**On Linux/Mac:**
```bash
cd backend

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**On Windows (PowerShell):**
```powershell
cd backend

# Create virtual environment
python -m venv venv

# Activate it (PowerShell)
.\venv\Scripts\Activate.ps1

# If you get an error, run this first:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
pip install -r requirements.txt
```

**On Windows (CMD):**
```cmd
cd backend
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

1. Copy the example file:
   ```bash
   # Linux/Mac
   cp .env.example .env
   
   # Windows
   copy .env.example .env
   ```

2. Edit `.env` and add your Mistral API key:
   ```
   MISTRAL_API_KEY=your_key_here
   ```

### Step 5: Train the ML Model

This creates the prediction model (takes 2-5 minutes):

```bash
# Make sure you're in backend/ with venv activated
python -m ml.train
```

You should see output like:
```
Training ML model...
Best model: RandomForest
F1 Score: 0.7072
Model saved to ml/artifacts/model.pkl
```

### Step 6: Build the RAG Database

This indexes the medical documents:

```bash
python scripts/rebuild_vectordb.py --clear
```

You should see:
```
Processed 5 documents → 10 chunks
Vector database is ready!
```

### Step 7: Run the Server

```bash
uvicorn app.main:app --reload
```

The server starts at: **http://localhost:8000**

### Step 8: Test the API (Swagger)

Open your browser and go to: **http://localhost:8000/docs**

This opens the **Swagger UI** where you can test all APIs:

1. **Test Prediction:**
   - Click `/api/v1/predict/predict`
   - Click "Try it out"
   - Enter: `{"symptoms": ["headache", "nausea", "fatigue"]}`
   - Click "Execute"

2. **Test RAG Explanation:**
   - Click `/api/v1/explain/{disease_name}`
   - Enter: `migraine`
   - Click "Execute"

3. **Test Chat:**
   - Click `/api/v1/chat/start`
   - Click "Execute" → Copy the `session_id`
   - Click `/api/v1/chat/message`
   - Enter session_id and message

---

## 🎨 Step 9: Run the Frontend (Web UI)

The frontend is a friendly web page so you don't need Swagger to use the project. **Keep the backend running in its own terminal**, then open a **second terminal** for the frontend.

### Step 9.1: Install the frontend

**On Linux/Mac:**
```bash
cd frontend
npm install
```

**On Windows (PowerShell or CMD):**
```powershell
cd frontend
npm install
```

> The first install can take 1–2 minutes. It downloads React, Vite, Tailwind, etc.

### Step 9.2: Start the frontend

**On Linux/Mac:**
```bash
npm run dev
```

**On Windows (PowerShell or CMD):**
```powershell
npm run dev
```

You will see something like:
```
VITE v5.x ready in 120 ms
  ➜ Local: http://localhost:5173/
```

Open **http://localhost:5173** in your browser. 🎉

### Step 9.3: How to use each page

The site has 4 pages in the top navigation. The theme is **black + beige**.

| Page | URL | What to do |
|------|-----|------------|
| **Home** | `/` | Landing page. Click any feature card to jump to the matching tool. |
| **Predict** | `/predict` | Type a word in the search box (e.g. `fever`) → click the symptom chips you want → click **Predict Disease** to see the top‑5 results, or **Explain (SHAP)** to see which symptoms pushed the model toward that disease. |
| **Chat** | `/chat` | The assistant greets you automatically. Describe how you feel, answer its follow‑up questions, then click **Get Diagnosis** when you are ready. The system extracts symptoms and runs a prediction. |
| **Explain** | `/explain` | Click any disease pill to read an evidence‑based explanation with **cited sources**, or type a free‑form question (e.g. *“What are early signs of diabetes?”*) and click **Ask**. |

> Tip: keep two terminals open — one for the backend (`uvicorn`) and one for the frontend (`npm run dev`). Closing either one stops that part.

---

## 🧪 Running Tests

```bash
# Make sure you're in backend/ with venv activated
pytest tests/ -v
```

Expected output:
```
44 passed, 1 warning
```

---

## 📚 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict/predict` | POST | Predict disease from symptoms |
| `/api/v1/predict/symptoms` | GET | List all valid symptoms |
| `/api/v1/chat/start` | POST | Start a chat session |
| `/api/v1/chat/message` | POST | Send message to chat |
| `/api/v1/chat/finalize/{id}` | POST | Get final prediction from chat |
| `/api/v1/explain/` | GET | List available diseases |
| `/api/v1/explain/{disease}` | GET | Get disease explanation with sources |
| `/api/v1/explain/question` | POST | Ask a medical question |

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'xxx'"
Make sure your virtual environment is activated:
```bash
# Linux/Mac
source venv/bin/activate

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat
```

### "MISTRAL_API_KEY not configured"
1. Make sure you have a `.env` file in the project root
2. Add: `MISTRAL_API_KEY=your_actual_key`

### "No model found"
Run the training script:
```bash
cd backend
python -m ml.train
```

### Windows: "Execution Policy" Error
Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "FileNotFoundError: diseases_symptoms.csv"
Download the dataset from Kaggle (see Step 2 above).

### Frontend: "npm: command not found" / "'npm' is not recognized"
Node.js is not installed. Install it from https://nodejs.org/ (LTS version) and reopen your terminal.

### Frontend: page loads but says "Network Error" / nothing happens
1. Make sure the backend is running on **http://localhost:8000** in another terminal.
2. Open the browser DevTools (F12) → Network tab and check that requests go to `localhost:8000/api/v1/...`.
3. CORS errors? The backend already allows `localhost:5173` by default — restart `uvicorn` after editing `.env`.

### Frontend: `npm run dev` fails with `ENOENT package.json`
You are in the wrong folder. Make sure you ran `cd frontend` first:
```bash
# Linux/Mac
cd /path/to/medcareai/frontend && npm run dev

# Windows
cd C:\path\to\medcareai\frontend
npm run dev
```

### Port 5173 or 8000 already in use
- **Linux/Mac:** find and kill the process: `lsof -i :5173` then `kill <PID>`
- **Windows (PowerShell):** `netstat -ano | findstr :5173` then `taskkill /PID <PID> /F`

---

## 🏗 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Backend | FastAPI (Python) | Web server + API |
| ML Model | RandomForest, XGBoost | Disease prediction |
| LLM | Mistral AI + LangChain | Chat + explanations |
| Vector DB | ChromaDB | Document search |
| Embeddings | sentence-transformers | Text → vectors |
| Frontend | React + Tailwind | Web interface |

---

## 📝 Adding More Medical Documents

To add more diseases to the RAG knowledge base:

1. Create a text file in `backend/rag/data/documents/`:
   ```
   TITLE: Disease Name
   SOURCE: WHO / Medical Organization
   URL: https://source-url.com
   
   Your content here...
   ```

2. Rebuild the vector database:
   ```bash
   python scripts/rebuild_vectordb.py
   ```

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Commit: `git commit -m "Add my feature"`
6. Push: `git push origin feature/my-feature`
7. Open a Pull Request

---

## ⚠️ Disclaimer

This is an **educational project**. It is NOT a substitute for professional medical advice. Always consult a healthcare provider for medical decisions.

---

## 📄 License

MIT License - feel free to use this project for learning!

---

## 🙋 Need Help?

- Open an issue on GitHub
- Check the Swagger docs at `/docs`
- Read the code comments

Happy coding! 🎉
