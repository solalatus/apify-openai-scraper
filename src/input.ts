import { ProxyConfigurationOptions, GlobInput, RequestOptions } from '@crawlee/core';

/**
 * Input schema in TypeScript format.
 */
export interface Input {
    startUrls: RequestOptions[];
    globs: GlobInput[];
    linkSelector?: string;
    openaiApiKey: string;
    openaiOrganizationId?: string;
    instructions: string;
    model?: string;
    targetSelector?: string;
    content?: string;
    maxPagesPerCrawl: number;
    maxCrawlingDepth: number;
    proxyConfiguration: ProxyConfigurationOptions;
    longContentConfig?: 'truncate' | 'split' | 'skip';
}

export const HTML_TAGS_TO_IGNORE = ['script', 'style', 'noscript'];
