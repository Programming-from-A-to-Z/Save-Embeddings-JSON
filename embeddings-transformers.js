import * as fs from 'fs';
import { pipeline } from '@xenova/transformers';

// Load the embeddings model
const extractor = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5');

// Read and split the transcript into chunks
console.log('Reading source file...');
const raw = fs.readFileSync('p5.txt', 'utf-8');
let chunks = raw.split(/\n+/);

// Trim each chunk and filter out empty strings
chunks = chunks.map((chunk) => chunk.trim()).filter((chunk) => chunk !== '');
console.log(`Total chunks to process: ${chunks.length}`);

// Start the process of generating embeddings
console.log('Starting embedding generation...');

// Array to store all embeddings in output JSON
const outputJSON = { embeddings: [] };

// Process chunks in batches
for (let i = 0; i < chunks.length; i++) {
  console.log(`Processing ${i + 1}/${Math.ceil(chunks.length)}`);

  // Generate the embedding from text
  const output = await extractor(chunks[i], {
    pooling: 'mean',
    normalize: true,
  });
  // Extract the embedding output
  const embedding = output.tolist()[0];

  // Add current batch embeddings
  outputJSON.embeddings.push({ text: chunks[i], embedding });
}

// Write the embeddings to a JSON file
const fileOut = 'embeddings.json';
fs.writeFileSync(fileOut, JSON.stringify(outputJSON));
console.log(`Embeddings saved to ${fileOut}`);
