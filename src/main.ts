import { Actor } from 'apify';
import { PlaywrightCrawler, Dataset, log } from 'crawlee';
import { createRequestDebugInfo } from '@crawlee/utils';
import { Input } from './input.js';
import {
    processInstructions,
    getNumberOfTextTokens,
    getOpenAIClient,
    validateGPTModel,
    rethrowOpenaiError,
    OpenaiAPIUsage,
} from './openai.js';
import {
    chunkTextByTokenLenght,
    htmlToMarkdown,
    htmlToText,
    shortsTextByTokenLength,
    shrinkHtml,
    tryToParseJsonFromString,
} from './processors.js';

const DEFAULT_OPENAI_MODEL = 'gpt-3.5-turbo';
const DEFAULT_CONTENT = 'markdown';

const MAX_REQUESTS_PER_CRAWL = 100;

const MERGE_DOCS_SEPARATOR = '----';

const MERGE_INSTRUCTIONS = `Merge the following text separated by ${MERGE_DOCS_SEPARATOR} into a single text. The final text should have same format.`;

await Actor.init();

const input = await Actor.getInput() as Input;

if (!input) throw new Error('INPUT cannot be empty!');

const openai = await getOpenAIClient(input.openaiApiKey, "");
const modelConfig = validateGPTModel(input.model || DEFAULT_OPENAI_MODEL);

const crawler = new PlaywrightCrawler({
    launchContext: {
        launchOptions: {
            headless: true,
        },
    },
    sessionPoolOptions: {
        blockedStatusCodes: [401, 429],
    },
    preNavigationHooks: [
        async ({ blockRequests }) => {
            await blockRequests();
        },
    ],
    requestHandlerTimeoutSecs: 3 * 60,
    proxyConfiguration: input.proxyConfiguration && await Actor.createProxyConfiguration(input.proxyConfiguration),
    maxRequestsPerCrawl: input.maxPagesPerCrawl || MAX_REQUESTS_PER_CRAWL,

async requestHandler({ request, page, enqueueLinks }) {
    const { depth = 0 } = request.userData;
    log.info(`Opening ${request.url}...`);

    const isDepthLimitReached = !!input.maxCrawlingDepth && depth < input.maxCrawlingDepth;
    if (input.linkSelector && input?.globs?.length && !isDepthLimitReached) {
        const enqueuedRequests = await enqueueLinks({
            selector: input.linkSelector,
            globs: input.globs,
            userData: {
                depth: depth + 1,
            },
        });

        const processedRequests = enqueuedRequests.map(req => req.url);

        const enqueuedLinks = processedRequests.filter(req => !req.wasAlreadyPresent);
        const alreadyPresentLinksCount = processedRequests.length - enqueuedLinks.length;
        log.info(
            `Page ${request.url} enqueued ${enqueuedLinks.length} new URLs.`,
            { foundLinksCount: enqueuedLinks.length, enqueuedLinksCount: enqueuedLinks.length, alreadyPresentLinksCount },
        );
    }

    const originalContentHtml = input.targetSelector
        ? await page.$eval(input.targetSelector, (el) => el.innerHTML)
        : await page.content();

    let pageContent = '';
    const content = input.content || DEFAULT_CONTENT;
    switch (content) {
        case 'markdown':
            pageContent = htmlToMarkdown(originalContentHtml);
            break;
        case 'text':
            pageContent = htmlToText(originalContentHtml);
            break;
        case 'html':
        default:
            pageContent = shrinkHtml(originalContentHtml);
            break;
    }
    const contentTokenLength = getNumberOfTextTokens(pageContent);
    const instructionTokenLength = getNumberOfTextTokens(input.instructions);

    let answer = '';
    const openaiUsage = new OpenaiAPIUsage(modelConfig.model);
    if (contentTokenLength > modelConfig.maxTokens) {
        if (input.longContentConfig === 'skip') {
            log.info(
                `Skipping page ${request.url} because content is too long for the ${modelConfig.model} model.`,
                { contentLength: pageContent.length, contentTokenLength, url: content },
            );
            return;
        } if (input.longContentConfig === 'truncate') {
            const contentMaxTokens = (modelConfig.maxTokens * 0.9) - instructionTokenLength; 
            const truncatedContent = shortsTextByTokenLength(pageContent, contentMaxTokens);
            log.info(
                `Processing page ${request.url} with truncated text using GPT instruction...`,
                { originalContentLength: pageContent.length, contentLength: truncatedContent.length, contentMaxTokens, contentFormat: content },
            );
            log.warning(`Content for ${request.url} was truncated to match GPT instruction limit.`);
            const prompt = `${input.instructions}\`\`\`${truncatedContent}\`\`\``;
            log.debug(
                `Truncated content for ${request.url}`,
                { promptTokenLength: getNumberOfTextTokens(prompt), contentMaxTokens, truncatedContentLength: getNumberOfTextTokens(truncatedContent) },
            );
            try {
                const answerResult = await processInstructions({ prompt, openai, modelConfig });
                answer = answerResult.answer;
                openaiUsage.logApiCallUsage(answerResult.usage);
            } catch (err: any) {
                throw rethrowOpenaiError(err);
            }
        } else if (input.longContentConfig === 'split') {
            const contentMaxTokens = (modelConfig.maxTokens * 0.9) - instructionTokenLength; 
            const pageChunks = chunkTextByTokenLenght(pageContent, contentMaxTokens);
            log.info(
                `Processing page ${request.url} with split text using GPT instruction...`,
                { originalContentLength: pageContent.length, contentMaxTokens, chunksLength: pageChunks.length, contentFormat: content },
            );
            const promises = [];
            for (const contentPart of pageChunks) {
                const prompt = `${input.instructions}\`\`\`${contentPart}\`\`\``;
                log.debug(
                    `Chunk content for ${request.url}`,
                    {
                        promptTokenLength: getNumberOfTextTokens(prompt),
                        contentMaxTokens,
                        truncatedContentPartLength: getNumberOfTextTokens(contentPart),
                        pageChunksCount: pageChunks.length,
                    },
                );
                promises.push(processInstructions({ prompt, openai, modelConfig }));
            }
            try {
                const results = await Promise.all(promises);
                answer = results.map((r) => r.answer).join('\n');
                for (const r of results) {
                    openaiUsage.logApiCallUsage(r.usage);
                }
            } catch (err: any) {
                throw rethrowOpenaiError(err);
            }
        } else {
            throw new Error(`Unsupported config value for longContentConfig: ${input.longContentConfig}`);
        }
    } else {
        log.info(`Processing page ${request.url} using GPT instruction...`, { contentLength: pageContent.length, contentFormat: content });
        const prompt = `${input.instructions}\`\`\`${pageContent}\`\`\``;
        log.debug(`Content for ${request.url}`, {
            promptTokenLength: getNumberOfTextTokens(prompt),
            pageContentTokenLength: getNumberOfTextTokens(pageContent),
        });
        try {
            const answerResult = await processInstructions({ prompt, openai, modelConfig });
            answer = answerResult.answer;
            openaiUsage.logApiCallUsage(answerResult.usage);
        } catch (err: any) {
            throw rethrowOpenaiError(err);
        }
    }

    const answerTokenLength = getNumberOfTextTokens(answer);
    log.info(`Received answer from ${modelConfig.model} model for ${request.url} with ${answerTokenLength} tokens.`);

    await Dataset.pushData({
        ...createRequestDebugInfo(request),
        contentLength: pageContent.length,
        contentTokenLength,
        instructionTokenLength,
        answerTokenLength,
        totalTokenLength: openaiUsage.getTotalTokenUsage(),
        openaiModel: modelConfig.model,
        openaiApiKey: openaiUsage.hasApiKeyExceeded(input.openaiApiKey),
        answer,
        originalContent: originalContentHtml,
        content: pageContent,
    });
},
});

await crawler.run();
