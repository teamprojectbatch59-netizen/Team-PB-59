# 🛡️ Hybrid Prompt Attack Detection and Mitigation Framework

## 📌 Overview

The **Hybrid Prompt Attack Detection and Mitigation Framework** is an AI-powered security system designed to protect Large Language Models (LLMs) from malicious prompt-based attacks. It detects and classifies threats such as Prompt Injection, Jailbreak, and Prompt Extraction using NLP and machine learning models.

The framework analyzes harmful prompts based on attack type, severity, and target component. It blocks malicious inputs and provides security recommendations while allowing safe prompts to be processed using Llama 3.2 3B.

## 🎯 Objectives

- Detect and distinguish between safe and harmful prompts.
- Classify harmful prompts into different attack categories.
- Analyze the severity and target components of detected attacks.
- Block harmful prompts and provide suitable mitigation recommendations.
- Generate safe responses for legitimate user prompts.

## 🚀 Key Features

- **Prompt Detection:** Identifies safe and harmful prompts using Prompt Guard 2.
- **Attack Classification:** Uses DeBERTa-v3, ModernBERT, and Llama Guard 3 for attack analysis.
- **Severity Analysis:** Evaluates the risk level of detected attacks.
- **Target Analysis:** Identifies the components targeted by malicious prompts.
- **Threat Mitigation:** Blocks harmful prompts and provides security recommendations.
- **Safe Response Generation:** Uses Llama 3.2 3B to generate responses for safe prompts.
- **Security Reporting:** Provides risk analysis results and security logs.

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development |
| Prompt Guard 2 | Safe and harmful prompt detection |
| DeBERTa-v3 | Attack classification |
| ModernBERT | Attack analysis |
| Llama Guard 3 | Security classification |
| Llama 3.2 3B | Safe response generation |
| NLP | Text preprocessing and analysis |
| Machine Learning | Threat detection and classification |

## ⚙️ System Workflow

1. User submits a prompt to the application.
2. The system preprocesses and analyzes the input.
3. Prompt Guard 2 identifies whether the prompt is safe or harmful.
4. Harmful prompts undergo attack classification, severity assessment, and target analysis.
5. Malicious prompts are blocked, and security recommendations are provided.
6. Safe prompts are forwarded to Llama 3.2 3B.
7. The generated response undergoes safety checks before being displayed.

## 📊 Expected Outcomes

- Accurate identification of harmful and safe prompts.
- Classification of different prompt-based attacks.
- Identification of attack severity and target components.
- Effective blocking of malicious prompts.
- Safe and appropriate response generation.
- Improved security and reliability of LLM applications.

## 🔐 Applications

- LLM-based chatbot security.
- AI application protection.
- Prompt injection detection.
- Jailbreak and prompt extraction prevention.
- Secure AI assistant development.

## 🎓 Project Information

**Project Type:** B.Tech Project Work-I

**Domain:** Artificial Intelligence, Machine Learning and Cybersecurity

**Project Focus:** Large Language Model Security

---

⭐ If you find this project useful, consider giving the repository a star!
