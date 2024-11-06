import Replicate from 'replicate';
import * as dotenv from 'dotenv';
import * as fs from 'fs';

dotenv.config();

// Initialize Replicate with API token
const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

// Model information for embedding
const version = 'b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305';
const model = 'replicate/all-mpnet-base-v25';

// Read and split the data into "chunks"
console.log('Reading source file...');
const raw = fs.readFileSync('data/p5.txt', 'utf-8');
let chunks = raw.split(/\n+/);

// Trim each chunk and filter out empty strings
chunks = chunks.map((chunk) => chunk.trim()).filter((chunk) => chunk !== '');
console.log(`Total chunks to process: ${chunks.length}`);

// Start the process of generating embeddings
console.log('Starting embedding generation...');
createEmbeddings(chunks);

// Function to create embeddings for each chunk
async function createEmbeddings(chunks) {
  // Number of chunks to process in each batch
  let batchSize = 10;

  // Array to store all embeddings
  let embeddings = [];

  // Process chunks in batches
  for (let i = 0; i < chunks.length; i += batchSize) {
    console.log(`Processing batch ${i / batchSize + 1}/${Math.ceil(chunks.length / batchSize)}`);
    const texts = chunks.slice(i, i + batchSize);
    const input = {
      text_batch: JSON.stringify(texts),
    };

    // Generate embeddings for the current batch
    const output = await replicate.run(`${model}:${version}`, { input });

    // Map each text to its corresponding embedding
    let currentEmbeddings = texts.map((text, index) => {
      const { embedding } = output[index];
      return { text, embedding };
    });

    // Add to the array
    embeddings = embeddings.concat(currentEmbeddings);
  }

  const jsonOut = { embeddings };
  // Write the embeddings to a JSON file
  const fileOut = 'embeddings/p5-embeddings.json';
  fs.writeFileSync(fileOut, JSON.stringify(jsonOut));
  console.log(`Embeddings saved to ${fileOut}`);
}
