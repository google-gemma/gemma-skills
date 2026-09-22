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

async function generate(question) {
    const messages = [
        {role: 'user', content: question}
    ];

    const prompt = generator.tokenizer.apply_chat_template(messages, {
        tokenize:false,
        add_generation_prompt: true,
    });

    // Collect streamed tokens; TextStreamer alone does not return text,
    // so without a callback the generation result would be discarded.
    let streamedText = '';
    const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true, // Don't stream the user's prompt back
        skip_special_tokens: true,
        callback_function: (chunk) => { streamedText += chunk; },
    });

    const output = await generator(prompt, {
        max_new_tokens: 256,
        streamer: streamer,
    });

    if (streamedText) {
        return streamedText;
    }

    // Fallback: derive the reply from the pipeline return value.
    const generated = Array.isArray(output) ? output[0]?.generated_text : output?.generated_text;
    if (typeof generated === 'string') {
        return generated.startsWith(prompt) ? generated.slice(prompt.length) : generated;
    }
    return '';
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

        console.log(await generate(question));

        console.log('\n');
    }
}

main().catch(err => {
    console.error('An error occurred:', err);
});