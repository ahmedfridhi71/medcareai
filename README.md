# MedCareAI

Advanced B2B/B2C medical decision-support platform combining supervised ML, LLM agents, RAG, and a full-stack web interface.

## Features

- **ML Disease Prediction**: RandomForest/XGBoost models with SHAP explainability
- **LLM Conversation Agent**: Natural symptom extraction using Mistral AI
- **RAG Scientific Agent**: Evidence-based explanations from medical literature
- **Recommendation Engine**: Lifestyle and treatment suggestions with safety checks
- **Role-Based Access**: Patient, Doctor, and Admin interfaces
- **Full-Stack Web App**: React frontend with FastAPI backend

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Backend | Python 3.11, FastAPI, SQLAlchemy (async), PostgreSQL, Redis |
| ML | scikit-learn, XGBoost, SHAP, MLflow |
| LLM/RAG | LangChain, Mistral AI, ChromaDB, BioBERT |
| Frontend | React (Vite), Tailwind CSS, React Query |
| Infra | Docker, GitHub Actions, Railway/Render |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL 15+

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/medcareai.git
cd medcareai

# Start all services with Docker
docker-compose up -d

# Or run locally:

# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your settings
```

## Project Structure

```
medcareai/
├── backend/           # FastAPI backend
│   ├── app/          # Application code
│   ├── ml/           # ML training and inference
│   ├── rag/          # RAG pipeline
│   └── tests/        # Backend tests
├── frontend/         # React frontend
└── docker-compose.yml
```

## API Documentation

Once running, access the API docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development Phases

- [x] Phase 0: Environment setup
- [ ] Phase 1: ML prediction model
- [ ] Phase 2: LLM conversation agent
- [ ] Phase 3: RAG scientific agent
- [ ] Phase 4: Recommendation + safety
- [ ] Phase 5: PostgreSQL database
- [ ] Phase 6: Authentication + RBAC
- [ ] Phase 7: Frontend patient side
- [ ] Phase 8: Frontend doctor side
- [ ] Phase 9: Docker + CI/CD
- [ ] Phase 10: Advanced AI features

## License

MIT License - See LICENSE file for details.

## Disclaimer

This is an academic/portfolio project. Not intended for real medical diagnosis. Always consult healthcare professionals for medical advice.
