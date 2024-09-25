import pprint

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig, AutoConfig
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import torch
import accelerate
import os
import jsonLoader as jsl
import keys

dirpath = os.path.dirname(__file__)
# open(filepath, 'r')
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeagueLLM:
    def __init__(self):
        self.docs = self.load_json_files("matchesjson/")
        self.llm = self.load_model()
        self.embeddings = self.embed_sentence_transformer()
        self.retriever = self.vstore_retriever(self.embeddings, self.docs)
        # self.keys = "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"
        # self.retriever = self.vstore_retriever()

    def load_model(self):
        #quantization configs
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        #load model
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path = model_id,
            token=keys.HF_TOKEN,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            config=model_config,
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=True,
            _fast_init=True,
            attn_implementation="flash_attention_2",
            token=keys.HF_TOKEN,
            # weights_only=True
        )

        # model.eval()
        # print(f"Model Loaded On: {device}")

        # use gpu
        # device = torch.device("cuda")
        # model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            clean_up_tokenization_spaces=True,
            token=keys.HF_TOKEN,
        )

        streamer = TextStreamer(tokenizer)

        mypipeline = pipeline(
            "text-generation",
            # "question-answering",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            streamer=streamer,
            return_full_text=True,
            temperature=0.6,
            max_new_tokens=500,
            repetition_penalty=1,
            do_sample=True
        )

        #attach pipeline to huggingface structure
        llm = HuggingFacePipeline(pipeline=mypipeline)


        return llm

    def load_json_files(self, path):
        texts = []
        for filenames in os.listdir(path):
            if filenames.endswith('.json'):
                #load json files
                json_file = os.path.join(path,filenames)
                loader = jsl.JSONLoader(json_file)
                docs = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                text = text_splitter.split_documents(docs)
                texts.append(text)
        return texts

    def embed_sentence_transformer(self):
        # Step 2: Create Embeddings
        model_name = "all-MiniLM-L12-v2"
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True }
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            # multi_process=True,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        return embeddings



    ##############################################################################

    def vstore_retriever(self, embeddings, docs):
        # print(docs)
        vectorstore = Chroma(collection_name="sample_collection", embedding_function=embeddings)
        for doc in docs:
            vectorstore.add_documents(doc)

        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 10, 'fetch_k': 300, 'score_threshold': 0.9, 'similarity_score_threshold': .9 })

        return retriever

    def llama3_chat(self):
        print("Hello!!!! I am llama3 and I can help with your document. \nIf you want to stop you can enter STOP at any point!")
        print()
        print("-------------------------------------------------------------------------------------")
        question = input()
        while question != "STOP":
            out = self.generate(question)
            print(out)
            print("\nIs there anything else you would like my help with?")
            print("-------------------------------------------------------------------------------------")
            question = input()

    def retrieve(self, question):
        print(question)
        docs = self.retriever.invoke(question)
        # print(docs)
        return "\n\n".join([d.page_content for d in docs])

    def augment2(self,question,context):
        return f"""
            <|begin_of_text|>
            <|start_header_id|>
              system
            <|end_header_id|>
               You are a training ai for league of legend gamers. 
               Don't provide commentary regarding the Context given. 
               Each line in the reply has no more than 85 characters.
               The given Context is compilation of match data from the top players in the world ranking.
               Summarize the Context and use it as comparison.
               It records per minute states of the games of the best players in the league.
               Every player has a participant Id and plays a champion with a name indicated as 
               championName, role, his kda or total number of kills, deaths and assists, team position and 
               teamid, 100 is team Blue and 200 is team Red.
               You also provide helpful tips regarding their questions to improve on the game.
               reply as follows:
                    [Compare kills, deaths, assists, and cs within the given game-time minute. 
                    Each line of text must not exceed 70 characters]                 
                    Recommended Items: 
                    [recommend 3 items appropriate for the given game-time minute]       
                    Tips to improve:
                    1. [Tip to improve]
                    2. [Tip to improve]
                    3. [Tip to improve]      
               For example:
                    Your stats are quite good for a 10 minute ADC:
                    Kills: 5 (avg 4.3), Deaths: 0 (avg 2.4), Assists: 1 (avg 4.3) 
                    CS: 97 (avg 71)
                    You're above average on kills, assists and CS. The average ADC has slightly more deaths than you.                                       
                    At 10 minutes these are the recommended items for ADC: 
                    Berserker's Greaves, Zeal, Blade of the Ruined King                    
                    To improve:
                    1. Focus on last-hitting minions to get more CS, especially in lane. 
                    2. Play more defensively and avoid fights if you're not ready. 
                    3. Position yourself well to help your jungler and engage at optimal times.
                    Overall, you're playing at a high level for your role and time! Keep up the good work.
            <|eot_id|>
            <|start_header_id|>
               user
            <|end_header_id|>
              Answer the user question based on data on the context and what's available online
              Context :{context}
              Question: {question}
            <|eot_id|>
            <|start_header_id|>
              assistant
            <|end_header_id|>"""

    def parse(self,string):
        return string.split("<|end_header_id|>")[-1]

    def generate(self,question, retriever_question):
        context = self.retrieve(retriever_question)
        prompt = self.augment2(question, context)
        answer = self.llm.invoke(prompt)
        return self.parse(answer)

