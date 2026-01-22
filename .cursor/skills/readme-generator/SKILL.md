---
name: readme-generator
description: This is a new rule
---

# Overview
Update the README.md in the root of this project. 

Guidelines:
- Title: Use the project/folder name or package.json "name".
- Description: Summarize purpose (from context or inferred).
- Features: Bullet-point list of main capabilities.
- Installation: Include backend (pip/Poetry) and frontend (npm/yarn/pnpm) steps if relevant.
- Usage: Show commands to run (python main.py, npm run dev, docker-compose up, etc.).
- Environment Variables: List keys only (no secret values).
- Deployment: Mention Docker if Dockerfile/compose is present, else skip.
- Architecture: Include a simple Mermaid diagram if services are present.
- Project Structure: Show a tree up to depth 2–3.
- License: Include if LICENSE is present.
- Keep it clear, professional, and concise.
- Include the usage of emoticons when proper.
- Code blocks: Always wrap commands, code samples, and diagrams in fenced code blocks 
    using triple backticks with the correct language identifier (e.g., ```bash```, ```python```, ```mermaid```).
- Never omit or truncate closing triple backticks.
- All installation and usage commands must appear in **copyable** fenced blocks (not inline).
- When showing folder trees or command-line examples, wrap them in ```bash``` or ```plaintext``` blocks.
- For Mermaid diagrams, always use the syntax ```mermaid\n{{diagram}}\n```.


Mermaid Diagram Rules (Strict):
1. Always start with `flowchart LR` or `flowchart TD`.
2. Node IDs must be simple alphanumeric tokens (A, B, API, DB).
3. Every node label must be wrapped in double quotes, no exceptions.
   Example: A["Frontend (React + Vite)"], B(("\"SQLite DB\""))
4. Edge labels must be quoted too: A -- "HTTP /api/*" --> B
5. Shapes:
   - Rectangle: A["My Label"]
   - Circle: A(("\"My Label\""))
   - Subgraph: subgraph FE["Frontend: React + Vite"] ... end
6. Never output [Label] without quotes if it contains spaces, punctuation, parentheses, slashes, or plus signs.
7. If in doubt, always quote the label.



Format the output as a valid Markdown README.md file.