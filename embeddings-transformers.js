import * as fs from 'fs';
import { pipeline } from '@huggingface/transformers';

// Load the embeddings model
const extractor = await pipeline('feature-extraction', 'mixedbread-ai/mxbai-embed-large-v1', {
  progress_callback: (x) => {
    if (x.progress) process.stdout.write(`\r${Math.round(x.progress)}% loaded`);
  },
});

// Read and split the transcript into chunks
console.log('\nReading source file...');
const raw = fs.readFileSync('data/p5.txt', 'utf-8');
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
  process.stdout.write(`\rProcessing ${i + 1}/${Math.ceil(chunks.length)}`);

  // Generate the embedding from text
  const output = await extractor(chunks[i], {
    pooling: 'cls',
  });

  // Extract the embedding output
  const embedding = output.tolist()[0];

  // Add current batch embeddings
  outputJSON.embeddings.push({ text: chunks[i], embedding });
}

// Write the embeddings to a JSON file
const fileOut = 'embeddings/p5-embeddings-tf.json';
fs.writeFileSync(fileOut, JSON.stringify(outputJSON));
console.log(`\nEmbeddings saved to ${fileOut}`);
