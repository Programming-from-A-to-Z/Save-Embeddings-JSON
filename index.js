import Replicate from 'replicate';
import * as dotenv from 'dotenv';
import * as fs from 'fs';

dotenv.config();

// Initialize Replicate with API token
const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

// Model information for embedding
const version = '9cf9f015a9cb9c61d1a2610659cdac4a4ca222f2d3707a68517b18c198a9add1';
const model = 'nateraw/bge-large-en-v1.5';

// Read and split the transcript into chunks
console.log('Reading source file...');
const raw = fs.readFileSync('p5.txt', 'utf-8');
let chunks = raw.split(/\n+/);
// Trim each chunk and filter out empty strings
chunks = chunks.map((chunk) => chunk.trim()).filter((chunk) => chunk !== '');
console.log(`Total chunks to process: ${chunks.length}`);

// Start the process of generating embeddings
console.log('Starting embedding generation...');
createEmbeddings(chunks);

// Function to create embeddings for each chunk
async function createEmbeddings(chunks) {
  let batchSize = 10; // Number of chunks to process in each batch
  let embeddings = []; // Array to store all embeddings

  // Process chunks in batches
  for (let i = 0; i < chunks.length; i += batchSize) {
    console.log(`Processing batch ${i / batchSize + 1}/${Math.ceil(chunks.length / batchSize)}`);
    const texts = chunks.slice(i, i + batchSize);
    const input = {
      texts: JSON.stringify(texts),
      batch_size: 32,
      convert_to_numpy: false,
      normalize_embeddings: true,
    };
    // Generate embeddings for the current batch
    const output = await replicate.run(`${model}:${version}`, { input });
    // Map each text to its corresponding embedding
    let currentEmbeddings = texts.map((text, index) => {
      return { text: text, embedding: output[index] };
    });
    // Add current batch embeddings
    embeddings = embeddings.concat(currentEmbeddings);
  }
  // Write the embeddings to a JSON file
  const fileOut = 'embeddings.json';
  fs.writeFileSync(fileOut, JSON.stringify(embeddings));
  console.log(`Embeddings saved to ${fileOut}`);
}
