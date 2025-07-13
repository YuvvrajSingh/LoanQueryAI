# 🚀 LoanQuery AI

A modern, beautiful Streamlit application that uses RAG (Retrieval Augmented Generation) to answer questions about loan approval patterns. Built with cutting-edge AI technologies and designed for ease of use. AI-Powered Loan Insight Assistant

A modern, beautiful Streamlit application that uses RAG (Retrieval Augmented Generation) to answer questions about loan approval patterns. Built with cutting-edge AI technologies and designed for ease of use.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 🎨 **Modern UI**: Beautiful gradient design with glass-morphism effects
- 🤖 **AI-Powered**: Real Groq API integration with demo mode fallback
- 🔍 **RAG Architecture**: Retrieval Augmented Generation for accurate responses
- 📊 **Smart Analytics**: Analyze loan approval patterns and trends
- 🔒 **Privacy First**: All processing happens locally on your machine
- 💡 **Interactive**: Sample questions and real-time chat interface
- 🚀 **Fast Setup**: One-command installation and setup

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  RAG Retriever   │───▶│   Groq LLM      │
│                 │    │                  │    │                 │
│ • Modern Design │    │ • FAISS Index    │    │ • LLaMA3-8B     │
│ • API Key Input │    │ • Embeddings     │    │ • Real-time     │
│ • Chat Interface│    │ • Similarity     │    │ • Demo Mode     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB+ RAM (for embedding models)
- **Training Dataset.csv** file (place in project root)

### 1. Installation

```bash
# Clone or download the project
git clone <repository-url>
cd LoanQuery-AI

# Place your Training Dataset.csv file in the project root directory

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Data

```bash
# Run one-time setup to prepare the data
python setup.py
```

This will:

- Load the Training Dataset.csv file from the project directory
- Download the embedding model (~90MB)
- Create FAISS vector index
- Process and store embeddings

### 3. Launch Application

```bash
# Start the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 4. Configure API Key (Optional)

**Option 1: Using Environment Variables (Recommended)**
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` file and add your Groq API key:
   ```
   GROQ_API_KEY=your_actual_groq_api_key_here
   USE_MOCK_RESPONSES=False
   ```
3. Get a free API key from [Groq Console](https://console.groq.com/)

**Option 2: Using the App Interface**
1. In the app, expand "⚙️ API Configuration"
2. Enter your API key
3. Uncheck "Use Demo Mode"
4. Click "Test Connection"

**Note**: The app works in demo mode without an API key, but real API integration provides better responses.

## 💬 Usage Examples

Try asking questions like:

- **"What factors affect loan approval rates?"**
- **"Why are some applications denied?"**
- **"How does income impact loan decisions?"**
- **"Are self-employed applicants treated differently?"**
- **"What's the approval rate by location?"**
- **"How important is credit history?"**

## 📊 Dataset Information

The application uses the **Training Dataset.csv** file which should contain loan approval data with these expected features:

| Feature             | Description               |
| ------------------- | ------------------------- |
| `Loan_ID`           | Unique identifier         |
| `Gender`            | Male/Female               |
| `Married`           | Marital status            |
| `Dependents`        | Number of dependents      |
| `Education`         | Graduate/Not Graduate     |
| `Self_Employed`     | Employment type           |
| `ApplicantIncome`   | Primary income            |
| `CoapplicantIncome` | Co-applicant income       |
| `LoanAmount`        | Requested amount          |
| `Credit_History`    | Credit history (0/1)      |
| `Property_Area`     | Urban/Semiurban/Rural     |
| `Loan_Status`       | Approved (Y) / Denied (N) |

**Note**: Place your `Training Dataset.csv` file in the project root directory before running the setup.

**Dataset Source**: You can download the Training Dataset.csv from [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction).

## �️ Technology Stack

| Component           | Technology                             |
| ------------------- | -------------------------------------- |
| **Frontend**        | Streamlit with custom CSS              |
| **AI Model**        | Groq LLaMA3-8B-Instruct                |
| **Embeddings**      | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector DB**       | FAISS (local, persistent)              |
| **Data Processing** | Pandas, NumPy                          |
| **Search**          | Cosine similarity                      |

## 📁 Project Structure

```
LoanQuery-AI/
├── Training Dataset.csv   # Loan dataset (place here)
├── app.py                 # Main Streamlit application
├── data_preprocessor.py   # Data processing & embedding
├── rag_retriever.py       # RAG retrieval system
├── llm_interface.py       # Groq API interface
├── setup.py              # One-time setup script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .venv/               # Virtual environment
├── faiss_index/         # Generated vector database
│   ├── loan_data.index  # FAISS index
│   └── metadata.pkl     # Metadata
└── __pycache__/         # Python cache
```

## 🎨 UI Features

- **Modern Design**: Gradient backgrounds with glass-morphism
- **Responsive Layout**: Works on desktop and mobile
- **Dark Theme**: Easy on the eyes
- **Smooth Animations**: Engaging user experience
- **Interactive Elements**: Hover effects and transitions
- **Professional Typography**: Inter font family

## 🔧 Configuration

### Environment Variables

You can set these environment variables instead of using the UI:

```bash
export GROQ_API_KEY="your_api_key_here"
```

### Customization

Edit `app.py` to customize:

- UI colors and styling
- Sample questions
- Response templates
- Layout components

## � Troubleshooting

### Common Issues

1. **"Training Dataset.csv not found"**

   - Ensure the `Training Dataset.csv` file is in the project root directory
   - Check the exact filename (case sensitive)

2. **"Module not found" errors**

   ```bash
   pip install -r requirements.txt
   ```

3. **"FAISS index not found"**

   ```bash
   python setup.py
   ```

4. **Slow performance**

   - First run downloads embedding model (~90MB)
   - Subsequent runs use cached models
   - Reduce `top_k` in retrieval for faster responses

5. **API connection issues**
   - Check your Groq API key
   - Verify internet connection
   - Use demo mode as fallback

### Performance Tips

- **Memory**: 4GB+ RAM recommended for optimal performance
- **Storage**: ~500MB needed for models and data
- **Network**: Only needed for initial model download and API calls

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

Need help? Here are your options:

- � Check this README for common solutions
- 🐛 Report bugs by opening an issue
- 💡 Request features via issues
- 📧 Contact the maintainers

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Groq](https://groq.com/) for fast LLM inference
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://faiss.ai/) for efficient vector search
- [Hugging Face](https://huggingface.co/) for the model ecosystem

---

⭐ **Star this repo if you found it helpful!** ⭐
