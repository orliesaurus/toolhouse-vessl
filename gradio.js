import express from 'express';
import { Client } from '@gradio/client';

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

let client;

async function setupGradioClient() {
  try {
    client = await Client.connect("https://run-execution-oe12r4vcmuaw-run-execution-7860.oregon.google-cluster.vessl.ai/");
    console.log("Connected to Gradio client");
  } catch (error) {
    console.error("Failed to connect to Gradio client:", error);
  }
}
app.get('/', async (req, res) => {
    res.json({ message: 'Hello, World!' });
});

app.post('/generate-image', async (req, res) => {
    if (!client) {
        return res.status(503).json({ error: 'Gradio client not connected' });
    }

    const {
        width = 128,
        height = 128,
        num_steps = 4,
        guidance = 3.5,
        seed = "-1",
        prompt = "An Astronaut",
    } = req.body;

    try {
        const result = await client.predict("/generate_image", {
            width,
            height,
            num_steps,
            guidance,
            seed,
            prompt,
        });

        res.json({ imageUrl: result.data[0].url });
    } catch (error) {
        console.error('Error generating image:', error);
        res.status(500).json({ error: 'Failed to generate image' });
    }
});

setupGradioClient().then(() => {
    app.listen(port, () => {
        console.log(`Microservice listening at port ${port}`);
    });
});
