# 🛡️ Hybrid Prompt Attack Detection and Mitigation Framework

## 📌 Overview

The **Hybrid Prompt Attack Detection and Mitigation Framework** is an AI-powered security system designed to protect Large Language Models (LLMs) from malicious prompt-based attacks. The framework uses Natural Language Processing (NLP), machine learning, and multiple AI models to identify, classify, and mitigate threats such as Prompt Injection, Jailbreak, and Prompt Extraction.

The system uses **ModernBERT as its primary prompt detection model**, supported by DeBERTa-v3, ModernBERT, and Llama Guard 3 for attack classification and security analysis. It evaluates detected threats based on attack category, severity level, and targeted components.

Malicious prompts are blocked, and appropriate security recommendations are provided. Safe prompts are forwarded to **Llama 3.2 3B**, which generates responses after passing through additional safety checks.

The framework aims to improve the security, reliability, and responsible usage of LLM-based applications.

## 🎯 Objectives

- Detect and distinguish between safe and harmful prompts.
- Identify different prompt-based attack categories.
- Classify attacks such as Prompt Injection, Jailbreak, and Prompt Extraction.
- Analyze the severity and target components of detected attacks.
- Block malicious prompts and provide suitable mitigation recommendations.
- Generate safe responses for legitimate user prompts using Llama 3.2 3B.
- Maintain security logs and generate threat analysis reports.

## 🚀 Key Features

- **Prompt Detection:** Uses ModernBERT as the primary model to identify safe and malicious prompts.
- **Attack Classification:** Uses DeBERTa-v3, ModernBERT, and Llama Guard 3 for analyzing and classifying prompt-based attacks.
- **Severity Analysis:** Evaluates the risk level of detected attacks.
- **Target Analysis:** Identifies the components or functionalities targeted by malicious prompts.
- **Threat Mitigation:** Blocks harmful prompts and provides suitable security recommendations.
- **Safe Response Generation:** Uses Llama 3.2 3B to generate responses for legitimate prompts.
- **Response Safety Checking:** Performs additional safety validation before displaying generated responses.
- **Security Reporting:** Maintains security logs and generates risk analysis reports.
- **User Authentication:** Supports secure user access through email-password authentication and SMTP-based OTP verification.

## 🧠 Technologies Used

| Technology / Model | Purpose |
|---|---|
| Python | Core development |
| ModernBERT | Primary prompt detection and attack analysis |
| DeBERTa-v3 | Prompt attack classification |
| Llama Guard 3 | Security classification and threat analysis |
| Llama 3.2 3B | Safe response generation |
| NLP | Text preprocessing and prompt analysis |
| Machine Learning | Threat detection and classification |
| SMTP | Email-based OTP verification |

## ⚙️ System Workflow

1. **User Authentication:** Users log in through email-password authentication and SMTP-based OTP verification.
2. **Prompt Submission:** The authenticated user submits a prompt to the application.
3. **NLP Preprocessing:** The input prompt is cleaned and prepared for analysis.
4. **Prompt Detection:** ModernBERT analyzes the input and identifies whether it is safe or potentially harmful.
5. **Attack Classification:** Harmful prompts undergo further analysis using DeBERTa-v3, ModernBERT, and Llama Guard 3.
6. **Severity and Target Analysis:** The system evaluates the attack category, severity level, and targeted component.
7. **Threat Mitigation:** Malicious prompts are blocked, and suitable security recommendations are generated.
8. **Safe Response Generation:** Legitimate prompts are forwarded to Llama 3.2 3B for response generation.
9. **Safety Verification:** Generated responses undergo additional safety checks before being displayed.
10. **Security Reporting:** The system records security events and generates risk reports.

## 📊 Expected Outcomes

- Accurate identification of safe and harmful prompts.
- Detection and classification of different prompt-based attacks.
- Identification of attack types, severity levels, and targeted components.
- Effective blocking of malicious prompts.
- Generation of suitable security recommendations.
- Safe response generation using Llama 3.2 3B.
- Improved security and reliability of LLM-based applications.
- Security logs and risk reports for threat monitoring.

## 🔐 Applications

- LLM-based chatbot security.
- AI assistant protection.
- Prompt injection detection and mitigation.
- Jailbreak attack detection.
- Prompt extraction prevention.
- Secure AI application development.
- AI-powered cybersecurity systems.

## 🛠️ Project Architecture

The framework follows a hybrid security architecture consisting of multiple detection, classification, and response-generation components.

| Component | Model / Technology |
|---|---|
| Primary Detection | ModernBERT |
| Attack Classification | DeBERTa-v3 |
| Additional Attack Analysis | ModernBERT |
| Security Classification | Llama Guard 3 |
| Response Generation | Llama 3.2 3B |
| Mitigation | Rule-based blocking and security recommendations |
| Reporting | Security logs and risk reports |

## 🎓 Project Information

- **Project Title:** A Hybrid Prompt Attack Detection and Mitigation Framework
- **Project Type:** B.Tech Project Work-I
- **Domain:** Artificial Intelligence, Machine Learning and Cybersecurity
- **Department:** Computer Science and Engineering
- **Specialization:** Artificial Intelligence and Machine Learning

## 🌱 Sustainable Development Goals (SDGs)

- **SDG 9 – Industry, Innovation and Infrastructure:** Promotes secure and innovative AI-based infrastructure.
- **SDG 16 – Peace, Justice and Strong Institutions:** Supports secure, trustworthy, and responsible digital systems.
- **SDG 4 – Quality Education:** Supports learning and research in AI, NLP, and cybersecurity.
- **SDG 8 – Decent Work and Economic Growth:** Improves the security and reliability of AI-based applications.
- **SDG 17 – Partnerships for the Goals:** Encourages responsible development and sharing of AI security practices.

## 🔮 Future Enhancements

- Improve detection accuracy using additional prompt attack datasets.
- Expand support for new and emerging prompt attack categories.
- Enhance severity prediction and threat intelligence.
- Integrate additional open-source security models.
- Improve detection performance and response safety.
- Develop advanced security dashboards and threat analytics.

---

**Built as part of B.Tech Project Work-I, focusing on LLM security, prompt attack detection, and AI-based threat mitigation.**
