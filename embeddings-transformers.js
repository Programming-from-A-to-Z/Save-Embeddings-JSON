import * as fs from 'fs';
import { pipeline } from '@xenova/transformers';

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
  // Load the embeddings model
  const bgeModel = await pipeline('feature-extraction', 'Supabase/bge-small-en');

  let embeddings = []; // Array to store all embeddings

  // Process chunks in batches
  for (let i = 0; i < chunks.length; i++) {
    console.log(`Processing batch ${i + 1}/${Math.ceil(chunks.length)}`);

    // Generate the embedding from text
    const output = await bgeModel(chunks[i], {
      pooling: 'mean',
      normalize: true,
    });
    // Extract the embedding output
    const embedding = Array.from(output.data);

    // Add current batch embeddings
    embeddings.push({ text: chunks[i], embedding });
  }

  const jsonOut = { embeddings };
  // Write the embeddings to a JSON file
  const fileOut = 'embeddings.json';
  fs.writeFileSync(fileOut, JSON.stringify(jsonOut));
  console.log(`Embeddings saved to ${fileOut}`);
}
