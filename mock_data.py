"""
Mock Database: Comprehensive AI & Machine Learning Knowledge Base
Structured for Progressive Context Loading.
"""

mock_database = {
    # LAYER 1: The Index
    "topics_index": [
        "artificial_intelligence",
        "traditional_machine_learning",
        "supervised_learning",
        "unsupervised_learning",
        "reinforcement_learning",
        "deep_learning",
        "natural_language_processing",
        "computer_vision",
        "generative_ai",
        "agentic_ai",
        "modern_machine_learning"
    ],
    
    # LAYER 2 & 3: Summaries and Deep Details
    "content": {
        "artificial_intelligence": {
            "summary": "Artificial Intelligence (AI) is the broad discipline of creating systems capable of performing tasks that typically require human intelligence, encompassing everything from rule-based systems to advanced neural networks.",
            "sections": {
                "ani_vs_agi": "Artificial Narrow Intelligence (ANI) is designed for specific tasks (e.g., chess bots, voice assistants). Artificial General Intelligence (AGI) is the theoretical milestone where an AI can understand, learn, and apply knowledge across any domain at a human level. Artificial Superintelligence (ASI) surpasses human capabilities.",
                "symbolic_ai": "Also known as GOFAI (Good Old-Fashioned AI), symbolic AI relies on human-readable representations of logic and rules, heavily used in expert systems during the 1980s before probabilistic machine learning took over.",
                "turing_test": "Proposed by Alan Turing in 1950, a test of a machine's ability to exhibit intelligent behavior indistinguishable from a human. While historically significant, modern AI evaluation relies more on specific benchmark datasets (like MMLU or HumanEval)."
            }
        },

        "traditional_machine_learning": {
            "summary": "Traditional Machine Learning refers to foundational algorithms that do not utilize deep neural networks. These models rely heavily on manual feature engineering and statistical mathematics.",
            "sections": {
                "linear_and_logistic_regression": "Linear regression predicts a continuous output based on input features by fitting a linear equation. Logistic regression is used for binary classification, mapping predictions to probabilities using a sigmoid function.",
                "decision_trees_and_forests": "Decision trees split data based on feature thresholds to make predictions. Random Forests are an ensemble method that builds multiple decision trees and merges them to get a more accurate and stable prediction, reducing overfitting.",
                "support_vector_machines": "SVMs find the optimal hyperplane that maximizes the margin between different classes in the dataset. They use the 'kernel trick' to project data into higher dimensions to solve non-linear classification problems."
            }
        },

        "supervised_learning": {
            "summary": "Supervised Learning is a paradigm where the model is trained on a labeled dataset, meaning each training example is paired with an output label. The goal is to learn a mapping function from inputs to outputs.",
            "sections": {
                "classification_tasks": "Classification assigns input data to distinct categories. Examples include spam detection (spam/not spam), image recognition (cat/dog), and medical diagnosis (benign/malignant).",
                "regression_tasks": "Regression models predict a continuous numerical value. Examples include predicting house prices based on square footage, forecasting stock prices, or estimating the age of a person from a photograph.",
                "overfitting_and_underfitting": "Overfitting occurs when a model learns the training data's noise and details too well, failing to generalize to new data. Underfitting happens when a model is too simple to capture the underlying trend of the data. Regularization (L1/L2) helps mitigate this."
            }
        },

        "unsupervised_learning": {
            "summary": "Unsupervised Learning deals with unlabelled data. The algorithm's objective is to discover hidden patterns, groupings, or structures within the dataset without explicit instructions on what to look for.",
            "sections": {
                "clustering_algorithms": "Clustering groups similar data points together. K-Means is a popular algorithm that partitions data into 'k' distinct clusters. DBSCAN is another method that groups dense regions of data, effectively handling outliers.",
                "dimensionality_reduction": "Techniques like Principal Component Analysis (PCA) and t-SNE reduce the number of random variables under consideration. This is crucial for visualizing high-dimensional data and speeding up the training of other ML models.",
                "anomaly_detection": "Identifying rare items, events, or observations which raise suspicions by differing significantly from the majority of the data. Widely used in credit card fraud detection and network security."
            }
        },

        "reinforcement_learning": {
            "summary": "Reinforcement Learning (RL) trains an agent to make a sequence of decisions in an environment to maximize a cumulative reward. It learns through trial and error.",
            "sections": {
                "markov_decision_processes": "MDPs are the mathematical framework for RL. They consist of States (S), Actions (A), Transition probabilities (P), and Rewards (R). The goal is to find an optimal Policy (π) that dictates the best action in every state.",
                "q_learning": "A model-free RL algorithm that seeks to learn the quality (Q-value) of actions, telling an agent what action to take under what circumstances. It updates a Q-table iteratively based on the Bellman equation.",
                "deep_reinforcement_learning": "Combines RL with deep neural networks. Deep Q-Networks (DQN) use neural nets to approximate the Q-value function, enabling agents to handle environments with massive state spaces, such as playing Atari games or Go (e.g., AlphaGo)."
            }
        },

        "deep_learning": {
            "summary": "Deep Learning is a subset of ML based on Artificial Neural Networks with multiple layers (deep networks). It excels at extracting high-level features from raw input, revolutionizing vision, speech, and text.",
            "sections": {
                "artificial_neural_networks": "ANNs consist of an input layer, hidden layers, and an output layer. Neurons apply weights, biases, and activation functions (like ReLU or Sigmoid) to their inputs to introduce non-linearity.",
                "backpropagation": "The core algorithm for training neural networks. It calculates the gradient of the loss function with respect to the weights by applying the chain rule of calculus backwards through the network, updating weights via Gradient Descent.",
                "convolutional_neural_networks": "CNNs use convolutional layers to automatically and adaptively learn spatial hierarchies of features. They are the primary architecture for image processing, utilizing filters to detect edges, textures, and complex objects.",
                "recurrent_neural_networks": "RNNs maintain an internal state (memory) to process sequences of inputs, making them ideal for time-series and speech. LSTM (Long Short-Term Memory) networks are a special kind of RNN designed to avoid the vanishing gradient problem."
            }
        },

        "natural_language_processing": {
            "summary": "NLP bridges the gap between human communication and computer understanding, enabling machines to read, parse, and generate human language.",
            "sections": {
                "tokenization_and_embeddings": "Tokenization splits text into smaller units (words or subwords). Embeddings (like Word2Vec or GloVe) convert these tokens into dense vectors of real numbers, capturing semantic meaning where words with similar meanings are mapped to nearby points in vector space.",
                "attention_mechanism": "Introduced to solve the bottleneck of fixed-length context vectors in RNNs. Attention allows a model to weigh the importance of different words in the input sequence dynamically when generating the output sequence.",
                "transformers": "The architecture introduced in the 'Attention Is All You Need' paper. It abandons recurrence entirely in favor of self-attention mechanisms, allowing for massive parallelization and serving as the foundation for modern LLMs."
            }
        },

        "generative_ai": {
            "summary": "Generative AI focuses on models that can create new, original content—such as text, images, audio, or code—by learning the underlying distribution of the training data.",
            "sections": {
                "large_language_models": "LLMs (like GPT-4, Claude, and Llama) are massive transformer networks trained on vast amounts of text. They predict the next token in a sequence, exhibiting emergent abilities like reasoning, summarization, and translation.",
                "retrieval_augmented_generation": "RAG is a technique to improve LLM accuracy and reduce hallucinations. Instead of relying solely on the LLM's parametric memory, RAG intercepts the user query, searches a vector database for factual context, and injects that context into the LLM prompt before generating an answer.",
                "diffusion_models": "Used primarily for image generation (e.g., Stable Diffusion, DALL-E). They work by adding Gaussian noise to training data and then learning to reverse the noising process to recover the data, eventually generating high-quality images from pure noise.",
                "prompt_engineering": "The practice of designing and refining input prompts to elicit optimal responses from Generative AI models. Techniques include Few-Shot prompting, Chain-of-Thought (CoT), and ReAct formatting."
            }
        },

        "agentic_ai": {
            "summary": "Agentic AI refers to systems where LLMs are used as reasoning engines to autonomously plan, execute tools, and iteratively solve complex problems without step-by-step human intervention.",
            "sections": {
                "autonomous_agents": "Unlike standard chatbots that just return text, an autonomous agent operates in a loop: it Observes a state, Reasons about what to do, Acts by calling a tool, and Observes the result until the overarching goal is met.",
                "tool_calling": "Also known as function calling. It allows an LLM to output a structured JSON payload that matches a specific tool's schema (e.g., 'search_web', 'execute_sql', 'send_email'). The application executes the tool and returns the data to the LLM.",
                "langchain_framework": "A popular open-source framework designed to simplify the creation of applications using LLMs. It provides standard interfaces for chains, document loaders, vector stores, and basic agent toolkits.",
                "langgraph_orchestration": "Built on top of LangChain, LangGraph represents agent workflows as highly controllable graphs (nodes and edges). It allows developers to build cyclic, stateful, multi-actor agent systems with built-in memory (checkpointers) and human-in-the-loop approvals."
            }
        },

        "modern_machine_learning": {
            "summary": "Modern Machine Learning encompasses the operationalization, optimization, and advanced deployment strategies of ML models in production environments.",
            "sections": {
                "mlops": "Machine Learning Operations (MLOps) is the practice of reliably and efficiently deploying and maintaining ML models in production. It includes automated CI/CD pipelines, model monitoring for data drift, and versioning of datasets and model artifacts.",
                "transfer_learning": "A research problem in ML that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. Pre-training a massive model (like BERT) and fine-tuning it on a smaller, specific dataset is a prime example.",
                "federated_learning": "A privacy-preserving ML technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchanging them. Only the model updates (weights) are sent to a central server."
            }
        }
    }
}