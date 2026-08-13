import uuid


class SessionManager:

    def __init__(self):

        self.sessions = {}


    # ==========================================
    # CREATE SESSION
    # ==========================================

    def create_session(self):

        session_id = str(
            uuid.uuid4()
        )

        self.sessions[session_id] = []

        return session_id


    # ==========================================
    # ADD MESSAGE
    # ==========================================

    def add_message(
        self,
        session_id,
        user_message,
        assistant_response
    ):

        if session_id not in self.sessions:

            self.sessions[session_id] = []


        self.sessions[session_id].append({

            "user": user_message,

            "assistant": assistant_response
        })


    # ==========================================
    # GET ONE SESSION HISTORY
    # ==========================================

    def get_history(
        self,
        session_id
    ):

        return self.sessions.get(
            session_id,
            []
        )


    # ==========================================
    # GET ALL SESSIONS
    # ==========================================

    def get_all_sessions(self):

        sessions = []

        for session_id, messages in self.sessions.items():

            # Default title
            title = "New Chat"

            # First user question as title
            if messages:

                title = messages[0].get(
                    "user",
                    "New Chat"
                )

                if len(title) > 35:

                    title = title[:35] + "..."


            sessions.append({

                "session_id": session_id,

                "title": title,

                "messages": messages
            })


        return sessions


    # ==========================================
    # DELETE SESSION
    # ==========================================

    def delete_session(
        self,
        session_id
    ):

        if session_id in self.sessions:

            del self.sessions[session_id]

            return True

        return False


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    manager = SessionManager()

    session = manager.create_session()

    print(
        "Session ID:",
        session
    )

    manager.add_message(
        session,
        "What is RAG?",
        "RAG is Retrieval Augmented Generation"
    )

    print(
        manager.get_history(
            session
        )
    )

    print(
        manager.get_all_sessions()
    )