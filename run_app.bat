@echo off
set PYTHONPATH=%PYTHONPATH%;%CD%
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
echo Starting TruthLens Pro Engines...
streamlit run app.py
