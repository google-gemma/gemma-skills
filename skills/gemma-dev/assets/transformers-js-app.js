import { pipeline, TextStreamer } from '@huggingface/transformers';
import cliProgress from 'cli-progress';
import inquirer from 'inquirer';

let generator;

async function initializeGemma() {
    console.log('Initializing Gemma model...');
    const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
    progressBar.start(100, 0);

    generator = await pipeline('text-generation', 'onnx-community/gemma-4-E2B-it-ONNX', {
        device: 'webgpu',
        dtype: 'q4',
        progress_callback: (progress) => {
            progressBar.update(progress.progress);
        },
    });

    progressBar.stop();
    console.log('Gemma model initialized!');
}

async function* generate(question) {
    const messages = [
        {role: 'user', content: question}
    ];

    const prompt = generator.tokenizer.apply_chat_template(messages, {
        tokenize:false,
        add_generation_prompt: true,
    });

    const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true, // Don't stream the user's prompt back
        skip_special_tokens: true,
    });

    await generator(prompt, {
        max_new_tokens: 256,
        streamer: streamer,
    });
}

async function main() {
    console.clear();
    await initializeGemma();

    while (true) {
        const { question } = await inquirer.prompt({
            type: 'input',
            name: 'question',
            message: "Ask Gemma anything:",
        });

        if (question.toLowerCase() === 'exit') {
            console.log('See you!');
            break;
        }

        console.log('\nGemma: ');

        for await (const chunk of generate(question)) {
            console.log(chunk);
        }

        console.log('\n');
    }
}

main().catch(err => {
    console.error('An error occurred:', err);
});