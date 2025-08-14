import {
  createOpenAI,
  type OpenAIProvider as VercelOpenAIProvider,
} from '@ai-sdk/openai';
import { embedMany } from 'ai';

import { metrics } from '../../../base';
import { CopilotProvider } from './provider';
import type {
  CopilotEmbeddingOptions,
  ModelConditions,
} from './types';
import { CopilotProviderType, ModelOutputType } from './types';

export type CustomConfig = {
  apiKey: string;
  baseURL?: string;
  model: string;
};

export class CustomProvider extends CopilotProvider<CustomConfig> {
  readonly type = CopilotProviderType.Custom;

  #instance!: VercelOpenAIProvider;

  override configured(): boolean {
    return !!this.config.apiKey && !!this.config.baseURL && !!this.config.model;
  }

  protected override setup() {
    super.setup();
    this.#instance = createOpenAI({
      apiKey: this.config.apiKey,
      baseURL: this.config.baseURL,
    });
    this.models = [
      {
        id: this.config.model,
        capabilities: [
          {
            input: [ModelInputType.Text],
            output: [ModelOutputType.Embedding],
            defaultForOutputType: true,
          },
        ],
      },
    ];
  }

  override async embedding(
    cond: ModelConditions,
    messages: string | string[],
    options: CopilotEmbeddingOptions = { dimensions: 256 }
  ): Promise<number[][]> {
    messages = Array.isArray(messages) ? messages : [messages];
    const fullCond = { ...cond, outputType: ModelOutputType.Embedding };
    await this.checkParams({ embeddings: messages, cond: fullCond, options });
    const model = this.selectModel(fullCond);

    try {
      metrics.ai
        .counter('generate_embedding_calls')
        .add(1, { model: model.id });

      const modelInstance = this.#instance.embedding(model.id);

      const { embeddings } = await embedMany({
        model: modelInstance,
        values: messages,
      });

      return embeddings.filter(v => v && Array.isArray(v));
    } catch (e: any) {
      metrics.ai
        .counter('generate_embedding_errors')
        .add(1, { model: model.id });
      throw this.handleError(e, model.id, options);
    }
  }

  private handleError(
    e: any,
    model: string,
    options: CopilotEmbeddingOptions = {}
  ) {
    console.error(e);
    throw new Error('custom embedding provider error');
  }
}
