10/04/2026:
    Rollbacks: remove traces off 2D-retro game and back to old idea (more heavy focus on ML/AI engineering)
    Focus on demo first, documentation follows later
    New quiz interface, responsible design, removed game aspects
    Decisions:
        When does the results screen appear?
            User clicks an explicit "See Results" button at any point: keeps it simple for the demo, alternatives are possible later (like after N questions)

    Known bugs:
        Only 1 question generated per request => very inneficient:
            In main.py:341 n_questions was forced to be 1 always
        Same question generated every time, next question is asked:
            In main.py:341 The retriever always uses request.zone (e.g., "RAG Architecture") as its search query. That string never changes, so vector similarity always returns the same top-k chunks → same LLM context → same question.

    Learned:
        The rule is: don't invoke the LLM until the user has committed to their intent since an LLM call is slow and expensive.
    
20/04/2026:
    Fix known bugs: only 1 question, same question every time
    Decisions:
        How to make sure that whenever a user asks for a quiz on a given topic, that the next quiz on the same topic is distinct from the first one?
            25 questions per zone + difficulty (e.g., "RAG Architecture" × "medium" = 25 Qs)
            Served 5 at a time as a quiz
            Level completion when the user passes all 5 quizzes (or hits a score threshold)
        Where to store the 25 generated questions:
            Sqlite database which is persistant and survives restarts
        Progress tracking:
            Server side: allowed implementation of UUID which will be useful in the future for progress tracking for example
            Player progress (player_progress table): two separate correct-answer counters (correct_count cumulative, current_quiz_correct scoped to current batch) because cumulative totals lose per-quiz granularity.
    
    Considerations:
        What's the latency of infering the LLM for 25 questions?
    
    Known issues:
        SOLVED: Frontend: there is no differentiation on the webpage between quizzes, it just says question x/x answered or correct, so the user has no idea that there are 5 or whatever quizzes
        SOLVED: The right answer is always answer A
            This was because of the prompt template: "answer": "A", ==> LLM puts the answer always in option A
    
    Learned:
        sqlite3 is Python stdlib, no new dependencies needed
        Backend generates 25 questions => Stores it in the frontend through sqlite, tradeoff:
            speed vs. freshness. You trade instant batch fetches for a question set that can go stale whenever the knowledge base changes. Mitigated by clearing the pool on every document upload.
        After restart, documents persist since they are uploaded to the ChromaDB and persist there, so when new documents get uploaded you get issues:
            Duplicate chunks get added to ChromaDB. The ingestion pipeline generates fresh random UUIDs every time (uuid.uuid4().hex in _run_ingestion), so ChromaDB treats re-uploaded content as brand new documents — no deduplication happens.
            It won't crash, but it inflates the collection and can bias retrieval toward over-represented content.
            Are the documents still identifiable within the vector database:
                 ChromaDB stores metadata alongside vectors, and LangChain's loaders automatically attach source (the file path) to every chunk's metadata. That means you can filter and delete by filename
