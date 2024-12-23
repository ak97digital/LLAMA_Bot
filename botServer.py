import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Function to scrape a website and extract textual data
def scrape_website(url):
    visited = set()
    data = []

    def scrape_page(page_url):
        if page_url in visited:
            return
        visited.add(page_url)

        try:
            response = requests.get(page_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
                page_content = ' '.join([para.get_text() for para in paragraphs])
                data.append(page_content)
                print(page_content)

                # Find all links on the page
                for link in soup.find_all('a', href=True):
                    full_url = urljoin(page_url, link['href'])
                    if url in full_url and full_url not in visited:
                        scrape_page(full_url)
        except requests.RequestException as e:
            print(f"Error accessing {page_url}: {e}")

    scrape_page(url)
    return ' '.join(data)

# Function to load a Hugging Face model
def load_model(model_name, token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
    return tokenizer, model

# Function to generate a response using the model
def generate_response(user_query, scraped_data, tokenizer, model):
    input_text = f"User: {user_query}\nData: {scraped_data}\nChatbot:"
    inputs = tokenizer.encode(input_text, return_tensors='pt')

    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response.split("Chatbot:")[-1].strip()

# Main function to run the chatbot
def main():
    website_url = ""
    hf_token = "" 
    print("Scraping website...")
    scraped_data = scrape_website(website_url)
    print("Scraping completed.")

    # Load the model
    model_name = "meta-llama/Llama-2-7b"
    print("Loading model...")
    tokenizer, model = load_model(model_name, token=hf_token)
    print("Model loaded.")

    # Chat loop
    print("Chatbot ready. Type 'exit' or 'quit' to end the conversation.")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        response = generate_response(user_query, scraped_data, tokenizer, model)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
