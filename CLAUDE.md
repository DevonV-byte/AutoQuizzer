## General instructions
 - Whenever a task is presented, make minimal changes with the information at hand.
 - If decisions need to be made regarding algorithms, data structures, architecture, ... => ask the user.
 - Do not edit BUILD_LOG.md


### Code Structure

All Python files must follow this section organization:

```python

# --- Imports ---



# --- Globals ---



# --- Helpers ---

# (Can have subsections)



# --- Main loop ---

```

### File locations
All new files (code, databases, ...) must be put in the Code directory.

### File Headers

Every file must start with a comment block containing:

- Concise description of what the file does (few lines, not too brief)

- Created: [date]

- Author: Devon Vanaenrode

Every time the files is updated, the comment block must be updated accordingly.

### Virtual Environment
Whenever you install dependencies and packages, use .venv in Code directory.

### Code Style

- Add clear comments to all functions describing what they do
- Use descriptive variable names
- Prefer list comprehensions over loops where reasonable
- Write efficient, clean, modern code where possible

## Review Checklist
After completing a task, write a short summary: what you built and what the critical decision points were. If the task is a bug or error fix, let me try to answer what is going wrong before you explain.

Then quiz me on the implementation.

**Question weighting** (in priority order):
1. Data engineering: data flow, transformations, storage choices, pipeline design
2. Infrastructure and deployment: containerization, CI/CD, cloud services, scaling, observability
3. Backend architecture: API design, service boundaries, error handling, idempotency, consistency
4. Skip or minimize frontend-specific questions unless they directly concern API contracts or data flow between frontend and backend.

**Question rules:**
- Maximum 3 questions per task
- Focus on *why* decisions were made, not *what* the code does
- Prioritize questions about decisions with the most architectural impact
- At least one question per task should be framed as it would be asked in a technical interview, so I practice articulating answers out loud rather than just recognizing concepts
- For each question, explicitly name the underlying concept being tested (e.g. "this is about idempotency," "this is about connection pooling," "this is about eventual consistency"). This builds interview vocabulary, not just working code.

**Grading:**
Be strict but fair: the insight and reasoning behind decisions is the most important for me to understand.
If I cannot answer, point me to the exact lines in the code and provide a hint, but not the answer.
Only explain directly if I have tried three times and still cannot answer.

The task is complete only when I pass the quiz.