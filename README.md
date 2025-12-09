# 🤖 Handwritten Digit Recognition — ML Web App

A production-ready **Machine Learning** web application for recognizing handwritten digits using a **CNN (Convolutional Neural Network)** trained on the MNIST dataset. Features real-time prediction, batch processing, and an interactive handwriting calculator powered by TensorFlow/Keras.

## 🎯 Machine Learning Features

- **Deep Learning Model**: Custom CNN architecture with 98% accuracy on MNIST test set
- **Real-time Inference**: Sub-100ms prediction latency for single digits
- **Batch Processing**: Parallel prediction on multiple images
- **Confidence Scoring**: Probability distribution over 10 classes (0-9)
- **Adaptive Preprocessing**: Smart image normalization and centering
- **Server-side ML Pipeline**: Gaussian blur smoothing, inversion, resizing
- **Safe Evaluation**: AST-based expression parser (no model injection attacks)

## 🧠 Model Architecture

**CNN with 4 Convolutional Layers:**
- Conv2D(64, 3×3) + ReLU
- Conv2D(32, 3×3) + MaxPool + ReLU
- Conv2D(16, 3×3) + MaxPool + ReLU
- Conv2D(64, 3×3) + MaxPool + ReLU
- Flatten → Dense(128) → Dense(10, Softmax)

**Performance:**
- Accuracy: 98%+ on MNIST test set
- Parameters: ~150K
- Training Time: ~5 minutes (10 epochs)
- Inference Speed: <100ms per image

## 📋 Requirements

- Python 3.8+
- TensorFlow/Keras
- Flask
- OpenCV
- Pillow
- NumPy

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/PoojithaAvvari/Handwritten-Digit-Recognition-DeepLearning-Web-App
cd Handwritten-Digit-Recognition-DeepLearning-Web-App
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Model (Optional)
If you don't have `trained_model.h5`:
```bash
python train.py
```

### 5. Run Application
```bash
python app.py
```

Open browser and navigate to `http://localhost:5000`

## 📁 Project Structure

```
Handwritten-Digit-Recognition/
├── app.py                    # Flask backend server
├── train.py                  # Model training script
├── load_model.py             # Single image prediction utility
├── tf_cnn.py                 # Legacy training script
├── requirements.txt          # Python dependencies
├── trained_model.h5          # Trained CNN model (not in repo)
├── templates/
│   └── index.html            # Main web UI
├── static/
│   ├── app.js                # Frontend JavaScript
│   └── style.css             # Stylesheet
├── uploads/                  # Uploaded images (not in repo)
└── README.md                 # This file
```

## 🏗️ Architecture

### Backend (Flask)
- **Model Loading**: Loads pre-trained TensorFlow model
- **Image Preprocessing**: Converts images to 28×28 MNIST format
- **Prediction Engine**: Returns digit + confidence score
- **Expression Evaluator**: Safe AST-based math evaluation

### Frontend (HTML/CSS/JavaScript)
- **Canvas Drawing**: Mouse/touch support for handwriting
- **API Communication**: Fetch requests to Flask endpoints
- **Gallery Management**: Browser localStorage persistence
- **Real-time Updates**: Immediate UI feedback

### Model
- **Architecture**: CNN with Conv2D, MaxPooling, Dense layers
- **Training Data**: MNIST (70,000 images, 10 classes: 0-9)
- **Accuracy**: ~98% on test set
- **Input**: 28×28 grayscale images
- **Output**: Softmax probability distribution over 10 digits

## 🎨 Usage Guide

### Drawing Mode
1. Click/drag on canvas to draw digit
2. Adjust confidence threshold (default: 0.7)
3. Enable/disable server smoothing
4. Click **Predict** to see result
5. Click **Save to Gallery** to store drawing

### Upload Mode
1. Select multiple image files
2. Adjust settings (threshold, smoothing)
3. Click **Predict All** for batch processing

### Calculator Mode
1. Draw digit on canvas
2. Click **Add Digit** to append prediction
3. Use operator buttons to build expression
4. Click **Evaluate** to calculate result

Example: Draw `5` → Click `+` → Draw `3` → Click `=` → Result: `8`

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main UI page |
| GET | `/model_status` | Check if model is loaded |
| POST | `/predict_draw` | Predict single drawn digit |
| POST | `/predict_files` | Batch predict uploaded images |
| POST | `/predict_and_append` | Predict and append to calculator |
| POST | `/append_symbol` | Add operator to expression |
| POST | `/clear_expression` | Reset calculator expression |
| GET | `/get_expression` | Retrieve current expression |
| POST | `/evaluate_expression` | Calculate math expression |
| POST | `/upload` | Handle file uploads |
| GET | `/list_uploads` | List uploaded files |
| DELETE | `/delete_file/<filename>` | Remove uploaded file |
| GET | `/uploads/<filename>` | Serve uploaded file |

## 🔒 Security Features

- **Safe Expression Evaluation**: AST-based parsing prevents code injection
- **Filename Sanitization**: Uses `werkzeug.utils.secure_filename()`
- **Path Traversal Prevention**: Restricted file access to `uploads/` directory
- **CSRF Token**: Flask sessions for state management

## ⚙️ Configuration

Edit these in `app.py`:

```python
MODEL_PATH = "trained_model.h5"    # Path to trained model
DEFAULT_THRESHOLD = 0.70           # Confidence threshold
APPLY_SMOOTHING = True             # Server-side Gaussian blur
UPLOAD_DIR = "uploads"             # Upload directory
```

## 📊 Model Training Details

### Dataset
- **Source**: TensorFlow MNIST (Yann LeCun)
- **Training**: 60,000 images
- **Testing**: 10,000 images
- **Classes**: 10 (digits 0-9)
- **Image Size**: 28×28 pixels, grayscale

### Architecture
```
Input (28×28×1)
  ↓
Conv2D (64 filters, 3×3) + ReLU
  ↓
Conv2D (32 filters, 3×3) + ReLU + MaxPool
  ↓
Conv2D (16 filters, 3×3) + ReLU + MaxPool
  ↓
Conv2D (64 filters, 3×3) + ReLU + MaxPool
  ↓
Flatten
  ↓
Dense (128) + ReLU
  ↓
Dense (10) + Softmax
  ↓
Output (digit class probabilities)
```

### Training
- **Optimizer**: Adam
- **Loss**: SparseCategoricalCrossentropy
- **Epochs**: 10
- **Batch Size**: 32
- **Expected Accuracy**: ~98%

## 🎬 Demo & Screenshots

### 1. Draw & Predict Digits
![Drawing Interface](assets/images/screenshot1.png)
*Real-time handwritten digit recognition with confidence scores*

### 2. Handwriting Calculator
![Calculator Interface](assets/images/screenshot2.png)
*Build math expressions by drawing digits and clicking operators*

---

## 🛠️ Troubleshooting

### "Model file not found"
- Run `python train.py` to generate `trained_model.h5`
- Or download a pre-trained model

### Low prediction accuracy
- Increase canvas size (currently 280×280)
- Enable server smoothing checkbox
- Draw more clearly in center of canvas
- Lower confidence threshold

### Port already in use
```bash
# Change port in app.py
app.run(debug=True, port=5001)
```

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Main Flask backend with model loading, prediction, expression evaluation |
| `train.py` | CNN model training on MNIST dataset |
| `load_model.py` | Standalone utility for single image prediction |
| `index.html` | Responsive web UI with drawing canvas & calculator |
| `app.js` | Frontend logic: canvas events, API calls, gallery management |
| `style.css` | Responsive styling with theme support |

## 🎓 Learning Resources

- [TensorFlow/Keras](https://www.tensorflow.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [CNN Fundamentals](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)

## 📄 License

MIT License - Feel free to use for personal/educational projects

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 👤 Author

Poojitha Avvari / [@PoojithaAvvari](https://github.com/PoojithaAvvari)

## 📧 Contact

Email: poojithaavvari2005@gmail.com

---

**Built with ❤️ for handwriting recognition**