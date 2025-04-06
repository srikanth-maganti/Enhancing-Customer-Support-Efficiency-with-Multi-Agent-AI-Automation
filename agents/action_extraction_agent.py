from autogen import Agent
from gen_model import run
class ActionExtractionAgent(Agent):
    def __init__(self, name="action_extraction", model="gemini-2.0-flash", max_tokens=500):
        super().__init__(name,model,max_tokens)
        self.title=name
        self.model = model
        self.max_tokens = max_tokens

    def extract_actions(self, conversation_summary):
        prompt = f"""
        You are an Action Extraction Agent for a technical customer support system.

        Your task is to read the following summarized customer conversation and extract only the specific action items that represent technical problems or tasks that need resolution.

        Guidelines:
        - Focus only on technical issues or requests that require action.
        - Ignore greetings, thank yous, small talk, or vague statements.
        - Each action should be a clear, concise problem statement or task.
        - Use short phrases or 1-line descriptions.

        Examples of valid actions:
        - Network connection issue
        - Software installation failure
        - Account synchronization bug
        - Payment gateway integration failure
        - Device compatibility error

        Conversation Summary:
        \"\"\"
        {conversation_summary}
        \"\"\"

        Return the list of action items, each on a new line.
        """

        response = run(prompt)
        return response

    def handle_message(self, message):
        """
        Process incoming messages and return extracted actions.
        """
        actions = self.extract_actions(message)
        print(actions)
        return {"agent": self.title, "actions": actions}

action_extracter = ActionExtractionAgent()
