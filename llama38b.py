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
from ignoreme import keys
import jsonLoader as jsl

dirpath = os.path.dirname(__file__)
# open(filepath, 'r')
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeagueLLM:
    def __init__(self):
        self.docs = self.load_json_files("./matchesjson/")
        self.llm = self.load_model()
        self.embeddings = self.embed_sentence_transformer()
        self.retriever = self.vstore_retriever(self.embeddings, self.docs)
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
        # model_id = "unsloth/llama-3-8b-bnb-4bit"
        # model_id = "C:/Users/LowQi/.cache/huggingface/hub/models--MaziyarPanahi--Llama-3-8B-Instruct-v0.10/snapshots/55a6fc03e04f1a68a5e2df16f3d0485d9ea357c8"
        # model_id = "vicgalle/Configurable-Llama-3.1-8B-Instruct"
        # model_id = "xxx777xxxASD/L3.1-ClaudeMaid-4x8B" # too big
        # model_id = "ValiantLabs/Llama3.1-8B-ShiningValiant2"
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # model_id = "vicgalle/Humanish-Roleplay-Llama-3.1-8B"
        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path = model_id,
            token=keys.HF_TOKEN
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
            token=keys.HF_TOKEN
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







    ##########################################################################

    # from langchain_community.llms import HuggingFacePipeline

    #Lets do a QuickTest
    # response = llm.invoke(messages)
    #
    # for x in response.split("\n"):
    #     print(x)

    # from langchain_core.prompts import PromptTemplate
    #
    # template = """Question: {question}
    #
    # Answer: The short answer is."""
    # prompt = PromptTemplate.from_template(template)
    #
    # chain = prompt | llm
    #
    # question = "what is the average kda of a silver ADC in league of legends in the first 5 minutes?"

    # print(chain.invoke({"question": question}))


    ##########################################################################

    def load_json_files(self, path):
        # path = os.path( matchesjson )
        # print(os.path.join(dirpath,path))
        texts = []
        for filenames in os.listdir(path):
            if filenames.endswith('.json'):
                # print(filenames)
                # fileList.append(filenames)
                #load json files
                # json_file_path = "matchesjson/EUW1_6999974885.json"
                json_file = os.path.join(path,filenames)
                loader = jsl.JSONLoader(json_file)
                docs = loader.load()
                # print(docs)
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                text = text_splitter.split_documents(docs)
                texts.append(text)
        # print(docs)
        # pprint.pprint(docs)
        return texts

    def embed_sentence_transformer(self):
        # Step 2: Create Embeddings
        # model_name = "all-mpnet-base-v2"  # Use your locally available Llama 3 model
        # model_name = "all-distilroberta-v1"
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
        # retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5})

        return retriever

    # def load_vstore(self):
        # print(docs)
        # persist_directory = "chromadb"
        # vectorstore = Chroma(collection_name="sample_collection",  embedding_function=self.embeddings)
        # vectorstore = self.vstore_doc_load(vectorstore, docs)
        # for doc in docs:
        #     vectorstore.add_documents(doc)
        # vectorstore.persist()
        # pprint.pprint(docs)
        # break
        # quit()

        # return retriever

    # pipe = Pipeline(llm,retriever)

    # #put docs in db. separated to prevent constant loading of same docs in db
    # def vstore_doc_load(self, vectorstore):
    #     # docs = self.docs
    #     for doc in self.docs:
    #         vectorstore.add_documents(doc)
    #
    #     return vectorstore
    #
    # def vstore_retriever(self):
    #     vectorstore = self.load_vstore()
    #     vectorstore = self.vstore_doc_load(vectorstore)
    #     retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 200, 'score_threshold': 0.9})
    #
    #     return retriever

    def llama3_chat(self):
        print("Hello!!!! I am llama3 and I can help with your document. \nIf you want to stop you can enter STOP at any point!")
        print()
        print("-------------------------------------------------------------------------------------")
        # pipe = Pipeline_custom(self.llm,self.retriever)
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
    def augment(self,question,context):
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
               Reply strictly in the following format and not comment on the Context: 
                [answer the question within 100 words] 
               Tips to improve:
                [Tips to improve].
               Sample reply is as follows:
                    In the given data, I'll compare your KDA and CS to the average KDA and CS of pro ADC 
                    players in the given data.
                    Average KDA of ADC players within 10 minutes and 20 seconds: 2.33/2.58/2.33
                    Average CS of ADC players within 10 minutes and 20 seconds: 54.47
                    Your KDA: 7/7/7
                    Your CS within 10 minutes and 20 seconds: 30
                    Your KDA is above the average, your CS is below the average.
                    
                    Tips to improve:
                    1. Focus on farming and getting more CS within the early game to increase your gold 
                        and item advantage.
                    2. Practice last-hitting minions to improve your CS and increase your gold income.
                    3. Improve your kill-to-death ratio by being more aggressive and taking fights when you      
            <|eot_id|>
            <|start_header_id|>
               user
            <|end_header_id|>
              Answer the user question based on data from https://www.op.gg/ context provided below
              Context :{context}
              Question: {question}
            <|eot_id|>
            <|start_header_id|>
              assistant
            <|end_header_id|>"""
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
        # prompt = self.augment(question, context)
        prompt = self.augment2(question, context)
        answer = self.llm.invoke(prompt)
        return self.parse(answer)
        # return answer


# question = "Im a mid laner with kda of 0/0/0 and 10 cs within 3 minutes, compare it to the average kda and cs of best players in the same role"
# myLLm = LeagueLLM()
# vstore = myLLm.load_vstore()
# myLLm.vstore_doc_load(vstore)
# answer = myLLm.generate(question)
# print(answer)
# answer = myLLm.vstore_retriever()
# myLLm.llama3_chat()
# get_answers("how many kills did participant 1 get?")
