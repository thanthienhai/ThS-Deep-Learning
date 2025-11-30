from huggingface_hub import list_repo_files

repo_id = "uit-nlp/vietnamese_students_feedback"
try:
    files = list_repo_files(repo_id, repo_type="dataset")
    print(f"Files in {repo_id}:")
    for f in files:
        print(f)
except Exception as e:
    print(f"Error: {e}")
