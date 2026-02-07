/**
 * Stubbed LLM for deterministic testing without real model.
 */

export interface FakeGenerateResult {
  text: string;
  tokensUsed: number;
}

export class FakeLlm {
  private generateResponse: string | Error = "<observation><type>decision</type><title>Test</title></observation>";
  private embedResponse: number[] = new Array(384).fill(0.1);
  private generateDelay: number = 0;

  setGenerateResponse(response: string | Error): void {
    this.generateResponse = response;
  }

  setEmbedResponse(embedding: number[]): void {
    this.embedResponse = embedding;
  }

  setGenerateDelay(ms: number): void {
    this.generateDelay = ms;
  }

  async generate(prompt: string, options?: { maxTokens?: number; temperature?: number }): Promise<FakeGenerateResult> {
    if (this.generateDelay > 0) {
      await new Promise((resolve) => setTimeout(resolve, this.generateDelay));
    }
    if (this.generateResponse instanceof Error) {
      throw this.generateResponse;
    }
    return { text: this.generateResponse, tokensUsed: 100 };
  }

  async embed(text: string): Promise<number[]> {
    return [...this.embedResponse];
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return texts.map(() => [...this.embedResponse]);
  }
}
