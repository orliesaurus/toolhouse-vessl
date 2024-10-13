import express from 'express';
import { Client } from '@gradio/client';

const app = express();
const port = process.env.PORT || 3000;

// Middleware to parse JSON bodies
app.use(express.json());

const client = await Client.connect("https://run-execution-oe12r4vcmuaw-run-execution-7860.oregon.google-cluster.vessl.ai/");

app.post('/generate-image', async (req, res) => {
    const {
        width = 128,
        height = 128,
        num_steps = 4,
        guidance = 3.5,
        seed = "-1",
        prompt = "An Astronaut",
    } = req.body;
    console.log("req", JSON.stringify(req.body));

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

app.listen(port, () => {
    console.log(`Microservice listening at port ${port}`);
});
