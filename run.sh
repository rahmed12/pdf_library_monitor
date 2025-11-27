python -m bookshelf_app.cli \
  --input-dir "/workspaces/audiobook-ai-docker-image/myProjects/booklib/inputbooks" \
  --pdf-output-dir "/workspaces/audiobook-ai-docker-image/myProjects/booklib/pdfs" \
  --ebook-output-dir "/workspaces/audiobook-ai-docker-image/myProjects/booklib/ebooks" \
  --default-model "gpt-oss:20b" \
  --metadata-model "gpt-oss:20b" \
  --classification-model "gpt-oss:20b" \
  --max-pages 10 \
  --once
