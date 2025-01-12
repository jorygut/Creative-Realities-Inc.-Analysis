import requests
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
import json
import time
from selenium.webdriver.chrome.options import Options
import os
from fuzzywuzzy import fuzz
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

def fetch_articles():
    API_KEY = "7c41beb12c4e426488da09b8200c61d5"
    BASE_URL = "https://newsapi.org/v2/everything"

    params = {
        "q": "Creative Realities Inc",
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 100,
        "apiKey": API_KEY
    }
    all_articles = []
    page = 1

    while True:
        print(f"Fetching page {page}...")
        params["page"] = page
        
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                articles = data.get("articles", [])
                if not articles:
                    break
                
                for idx, article in enumerate(articles, start=1 + (page - 1) * 100):
                    return_dict = {
                        "title": article.get('title'),
                        "source": article.get('source', {}).get('name'),
                        "publishedAt": article.get('publishedAt'),
                        "url": article.get('url')
                    }
                    all_articles.append(return_dict)
                    print(f"{idx}. {article.get('title')}")
                    print(f"   Source: {article.get('source', {}).get('name')}")
                    print(f"   Published At: {article.get('publishedAt')}")
                    print(f"   URL: {article.get('url')}\n")
                
                page += 1
            else:
                print("No articles found or an error occurred in the API response.")
                break
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            break

    print(f"\nTotal articles retrieved: {len(all_articles)}")
    return all_articles

def scrape_articles(articles):
    driver = webdriver.Chrome()
    for a in articles:
        try:
            driver.get(a['url'])

            page_text = driver.find_element(By.TAG_NAME, 'body').text
            a['text'] = page_text
        except:
            a['text'] = ""
    return articles

def chunk_text(text):
    chunks= []
    i = 0
    end_text = False
    while not end_text:
        cur_chunk = text[i:(i + 500)]
        chunks.append(cur_chunk)
        if len(cur_chunk) < 500:
            break
        i += 500
    return chunks

def analyze_text_emotions(articles):
    nltk.download('vader_lexicon')

    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    for a in articles:
        all_scores = []
        chunked_text = chunk_text(a['text'])
        for c in chunked_text:
            score = classifier(c)
            all_scores.append(score)
        a['scores'] = all_scores

    for art in articles:
        total_dict = {}
        for i in art['scores']:
            for l in i[0]:
                if l['label'] in total_dict:
                    total_dict[l['label']] += l['score']
                    total_dict[f'{l["label"]}_count'] += 1
                else:
                    total_dict[l['label']] = l['score']
                    total_dict[f'{l["label"]}_count'] = 1
        avg_dict = {}
        for t in total_dict:
            if "count" not in t:
                avg_dict[t] = total_dict[t] / total_dict[f'{t}_count']
        art.pop('scores', None)
        art.update(avg_dict)

    df = pd.DataFrame(articles)
    return df
    


def main():
    articles = fetch_articles()
    articles_with_text = scrape_articles(articles)
    df_analyzed = analyze_text_emotions(articles_with_text)
    df_analyzed.to_csv('create_realities_articles.csv')

if __name__ == "__main__":
    main()
