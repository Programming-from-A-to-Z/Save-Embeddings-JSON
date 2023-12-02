# Saving Embeddings to JSON file

## Overview

This is an example Node.js application processes a text corpus, generates embeddings for "chunks", and saves the embeddings to a local file. The embeddings can be used in another application (like a [Retrieval Augmentated Generation system](https://github.com/Programming-from-A-to-Z/Example-RAG-Replicate) or [2D/3D clustering demonstration using UMAP dimensionality reduction](https://editor.p5js.org/a2zitp/sketches/p63QTp0Sd))

- `index.js`: Process a text file and generate embeddings.
- `embeddings.json`: Precomputed embeddings generated from the text corpus.
- `.env`: API token

![A map of clustered p5.js function names](clustering.png)

## References

- [Using open-source models for faster and cheaper text embeddings](https://replicate.com/blog/run-bge-embedding-models)

## How-To

1. Install Dependencies

```sh
npm install
```

2. Set up the `.env` file with your Replicate API token:

```env
REPLICATE_API_TOKEN=your_api_token_here
```

3. Generate the `embeddings.json` file by running `index.js`. (You'll need to hard-code a text filename and adjust how the text is split up depending on the format of your data.)

```js
const raw = fs.readFileSync('text-corpus.txt', 'utf-8');
let chunks = raw.split(/\n+/);
```

```sh
node index.js
```
