## Instrutions
1. Install the required libraries:
```bash
pip install -r requirements.txt
```
2. Run the vectordb to create the database:
```bash
python vectordb.py --docs_path <path_to_your_doc> 
```
3. Run the main.py to start the chatbot:
```bash
python rag_chatbot.py
```
Utilzed command line interface to interact with the chatbot.
Note: The supported document is expected in pdf format.
The pdf document use in this repo is generated 