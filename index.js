const express = require('express');
const crypto = require('crypto');
const axios = require('axios');
const tmp = require('tmp');
const fs = require('fs');
const { Pinecone } = require('@pinecone-database/pinecone');
const { Document } = require('@langchain/core/documents');
const { ChatGoogleGenerativeAI } = require('@langchain/google-genai');
const { HuggingFaceInferenceEmbeddings } = require('@langchain/community/embeddings/hf');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { loadQAStuffChain } = require('langchain/chains');
const { PromptTemplate } = require('@langchain/core/prompts');
const { PineconeStore } = require('@langchain/pinecone');
require('dotenv').config();

// === Environment Setup ===
const GEMINI_API_KEY = process.env.GOOGLE_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = "hackrx-rag-index";
const HUGGING_FACE_API_KEY = process.env.HUGGING_FACE_API_KEY;

// === Express App & Security ===
const app = express();
app.use(express.json());
const cors = require('cors');
app.use(cors());

// === Logger Setup ===
const logger = {
    info: (message) => console.log(`[INFO] ${message}`),
    warn: (message) => console.warn(`[WARN] ${message}`),
    error: (message, trace) => console.error(`[ERROR] ${message}`, trace),
};

// === Language Model & Embeddings Setup ===
let embedding_model;
const getEmbeddingModel = async () => {
    if (!embedding_model) {
        embedding_model = new HuggingFaceInferenceEmbeddings({
            apiKey: HUGGING_FACE_API_KEY,
            model: "sentence-transformers/all-MiniLM-L6-v2",
        });
    }
    return embedding_model;
};

const llm = new ChatGoogleGenerativeAI({
    apiKey: GEMINI_API_KEY,
    model: "gemini-1.5-flash",
    temperature: 0.2,
});

// === QA Prompt Template ===
const qaPrompt = new PromptTemplate({
    inputVariables: ["context", "question"],
    template: `
You are an expert AI legal assistant specializing in Indian insurance policies.
Your task is to answer the user's question with extreme precision based ONLY on the provided context from the policy document.
If no context is provided, answer the question based on your own knowledge. Dont use \n in output response.
*Instructions:*
1.  *Strictly Contextual:* Your answer MUST be derived exclusively from the text provided in the 'Context' section. Do not use any external knowledge.
2.  *Concise and Formal:* Provide a direct, professional answer. Limit your response to 1-3 clear sentences.
3.  *No Fillers:* Do not include phrases like "According to the policy...", "The context states...", or any disclaimers.
4.  *Focus on Key Details:* Extract specific details such as waiting periods, monetary limits, eligibility criteria, and exclusions when relevant.

*Context:*
{context}

*Question:*
{question}

*Answer:*
`,
});

// === Pinecone Setup ===
let pinecone;
let pineconeIndex;

const initPinecone = async () => {
    if (PINECONE_API_KEY) {
        pinecone = new Pinecone({
            apiKey: PINECONE_API_KEY,
        });
        pineconeIndex = pinecone.Index(PINECONE_INDEX_NAME);
        logger.info(`Successfully connected to Pinecone index: ${PINECONE_INDEX_NAME}`);
    } else {
        logger.warn("PINECONE_API_KEY not found. Pinecone integration is disabled.");
    }
};
initPinecone();


// === Helper Functions ===
const getDocumentHash = (url) => {
    return crypto.createHash('sha256').update(url).digest('hex');
};

// === API Endpoints ===
app.get("/api", (req, res) => {
    res.json({
        status: "ok",
        message: "Welcome to the Policy Q&A API!",
        endpoints: {
            process: "POST /api/hackrx/process-document",
            query: "POST /api/hackrx/query"
        }
    });
});

// Endpoint to process and index a document. This is a long-running task.
app.post("/api/hackrx/process-document", async (req, res) => {
    const { docUrl } = req.body;
    if (!docUrl) {
        return res.status(400).json({ detail: "Missing 'docUrl' in request body." });
    }

    let pdfPath = null;
    try {
        const docHash = getDocumentHash(docUrl);
        logger.info(`Processing request for document hash: ${docHash}`);

        if (!pineconeIndex) {
            return res.status(500).json({ detail: "Pinecone is not configured." });
        }

        const stats = await pineconeIndex.describeIndexStats();
        if (stats.namespaces && stats.namespaces[docHash]) {
            logger.info(`Document hash ${docHash} already exists. Skipping processing.`);
            return res.status(200).json({ message: "Document already processed." });
        }

        logger.info(`Document not found in Pinecone. Processing and indexing: ${docUrl}`);
        const response = await axios.get(docUrl, { responseType: 'arraybuffer' });
        if (!response.headers['content-type'].startsWith("application/pdf")) {
            return res.status(400).json({ detail: "Invalid content type. URL must point to a PDF." });
        }

        const tmpFile = tmp.fileSync({ suffix: ".pdf" });
        pdfPath = tmpFile.name;
        fs.writeFileSync(pdfPath, response.data);

        const { PDFLoader } = require("langchain/document_loaders/fs/pdf");
        const loader = new PDFLoader(pdfPath);
        const documents = await loader.load();
        const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 150 });
        const chunks = await splitter.splitDocuments(documents);

        logger.info(`Loaded and split into ${chunks.length} chunks. Creating embeddings...`);
        const embeddingModel = await getEmbeddingModel();

        await PineconeStore.fromDocuments(chunks, embeddingModel, {
            pineconeIndex,
            namespace: docHash,
            maxConcurrency: 5,
        });

        logger.info("Indexing complete.");
        res.status(201).json({ message: "Document processed and indexed successfully.", docHash });

    } catch (error) {
        logger.error(`An unexpected error occurred during document processing: ${error.message}`, error);
        res.status(500).json({ detail: `An internal server error occurred: ${error.message}` });
    } finally {
        if (pdfPath) {
            fs.unlinkSync(pdfPath);
            logger.info("Temporary PDF file cleaned up.");
        }
    }
});


// Endpoint to ask questions about a pre-processed document.
app.post("/api/hackrx/query", async (req, res) => {
    const { docUrl, questions } = req.body;
    if (!docUrl || !questions || !Array.isArray(questions)) {
        return res.status(400).json({ detail: "Request body must include 'docUrl' and an array of 'questions'." });
    }

    try {
        const docHash = getDocumentHash(docUrl);
        logger.info(`Querying document hash: ${docHash}`);

        if (!pineconeIndex) {
            return res.status(500).json({ detail: "Pinecone is not configured." });
        }

        const stats = await pineconeIndex.describeIndexStats();
        if (!stats.namespaces || !stats.namespaces[docHash]) {
            return res.status(404).json({
                detail: "Document not found. Please process it first via the /api/hackrx/process-document endpoint."
            });
        }

        const qaChain = loadQAStuffChain(llm, "stuff", { prompt: qaPrompt });
        const answerList = [];
        const embeddingModel = await getEmbeddingModel();
        const vectorStore = await PineconeStore.fromExistingIndex(embeddingModel, {
            pineconeIndex,
            namespace: docHash,
        });

        for (const q of questions) {
            logger.info(`Answering question: ${q}`);
            const relevantDocs = await vectorStore.similaritySearch(q, 5);

            if (relevantDocs.length === 0) {
                logger.warn(`No relevant documents found for question: ${q}`);
                answerList.push("No relevant information found in the document to answer this question.");
                continue;
            }

            const result = await qaChain.invoke({ input_documents: relevantDocs, question: q });
            answerList.push(result.text.trim());
        }

        logger.info("All questions processed successfully.");
        res.json({ answers: answerList });

    } catch (error) {
        logger.error(`An unexpected error occurred during query: ${error.message}`, error);
        res.status(500).json({ detail: `An internal server error occurred: ${error.message}` });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    logger.info(`Server is running on port ${PORT}`);
});

module.exports = app;
