# RNN-final-project

## Download Steps
Download the model from : [Google Drive Link](https://drive.google.com/drive/folders/14uaWo6KFr4Agwti33J-8RLykP61mJOTw?usp=sharing)

## Place the model directly to the project root directory:
Place the model files to the current directory
After that, your directory structure should look like:
rnn-final-project/
├── analysis_results_toxicity=0.3/
├── analysis_results_toxicity=0.5/
├── base_results/
├── results_outputs/
├── toxigen_model/               
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── README.md
├── Toxigen_DeBerta_v3.ipynb
├── enhanced_defense_system.py
└── ... (other files)

## Run 
```bash
$ pip install streamlit
$ python -m streamlit run streamlit_app.py
```
