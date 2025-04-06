from autogen import Agent
from gen_model import run

class ConversationSummarizerAgent(Agent):
    def __init__(self, name="conversation_summarizer", model="gemini-2.0-flash", max_tokens=500):
        super().__init__(name)

        self.model = model
        self.max_tokens = max_tokens

    def summarize_conversation(self, conversation_history):
        """
        Summarizes a given conversation history using an LLM model.

        Args:
            conversation_history (list of dict): A list of message dictionaries containing 'role' and 'content'.

        Returns:
            str: A summarized version of the conversation.
        """
        prompt = f"""
        Summarize the following customer support conversation:
        {conversation_history}

        Provide a concise summary capturing the main issue, customer concerns.
        """

        response =run(prompt)
        return response

    def handle_message(self, message):
        """
        Process incoming messages and return a summary.
        """
        summary = self.summarize_conversation(message)
        return {"agent": self.name, "summary": summary}


summarizer = ConversationSummarizerAgent()

