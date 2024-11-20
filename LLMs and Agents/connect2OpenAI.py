from openai import OpenAI
import os
from datetime import datetime

def load_prompt_context():
    try:
        with open('Prompt1.txt', 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print("Warning: Prompt1.txt not found. Proceeding without context.")
        return ""

def save_conversation(question, response):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('conversation.txt', 'a', encoding='utf-8') as file:
        file.write(f"\n{'='*50}\n")
        file.write(f"Timestamp: {timestamp}\n")
        file.write(f"Question: {question}\n")
        file.write(f"Response: {response}\n")

def ask_openai():
    try:
        # Initialize the client
        client = OpenAI()
        
        # Load context from Prompt1.txt
        context = load_prompt_context()
        
        # Initialize messages with context if available
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        
        print("\nContext loaded successfully!" if context else "\nNo context loaded.")
        
        while True:
            # Get user input
            print("\nEnter your question (or 'quit' to exit, 'history' to view conversation.txt):")
            user_question = input("> ")

            # Check commands
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
                
            elif user_question.lower() == 'history':
                try:
                    with open('conversation.txt', 'r', encoding='utf-8') as file:
                        print("\nConversation History:")
                        print(file.read())
                except FileNotFoundError:
                    print("\nNo conversation history found.")
                continue

            # Add user question to messages
            messages.append({"role": "user", "content": user_question})

            # Make the API call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )

            # Get the response
            ai_response = response.choices[0].message.content

            # Add AI response to messages for context continuity
            messages.append({"role": "assistant", "content": ai_response})

            # Print the response
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
            print("Question:", user_question)
            print("\nResponse:", ai_response)
            print("\n" + "="*50)  # Separator for readability

            # Save the conversation to conversation.txt
            save_conversation(user_question, ai_response)

    except Exception as e:
        print(f"\nError type: {type(e)}")
        print(f"Error message: {str(e)}")

if __name__ == "__main__":
    print("Welcome to OpenAI Chat!")
    print("You can ask any question, and type 'quit' to exit.")
    print("="*50)
    ask_openai()
