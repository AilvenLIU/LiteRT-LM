/**
 * Copyright 2026 The ODML Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {LlmChatStateController} from '../state_controller.js';
import type {StoredMessage} from '../state_controller.js';
import {sharedStyles} from '../styles/shared_styles.js';

/* tslint:disable:no-new-decorators */

/** Component representing a single chat message bubble. */
@customElement('litert-chat-bubble')
export class LitertChatBubble extends LitElement {
  @property({ type: Object })
  message!: StoredMessage;

  @property({ type: Number })
  index!: number;

  @property({ type: Object })
  state!: LlmChatStateController;

  static override styles = [
    sharedStyles,
    css`
      :host {
        display: flex;
        flex-direction: column;
        width: 100%;
      }

      .message-bubble {
        display: flex;
        flex-direction: column;
        max-width: 75%;
        padding: 12px 16px;
        border-radius: 12px;
        font-size: 0.9375rem;
        line-height: 1.5;
        word-wrap: break-word;
        animation: fadeIn 0.2s ease-out;
        position: relative;
      }

      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
      }

      .message-bubble.user {
        align-self: flex-end;
        background-color: var(--slate-bubble-user);
        border: 1px solid var(--border);
        border-bottom-right-radius: 2px;
      }

      .message-bubble.assistant {
        align-self: flex-start;
        background-color: var(--slate-bubble-bot);
        border: 1px solid var(--border);
        border-bottom-left-radius: 2px;
      }

      .message-sender {
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }

      .message-sender.user {
        color: var(--teal);
        text-align: right;
      }

      .message-sender.assistant {
        color: var(--blue);
      }

      .message-content {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .message-user-text {
        white-space: pre-wrap;
      }

      .message-text {
        white-space: pre-wrap;
      }

      /* Thought process block */
      .thought-details {
        background-color: rgba(255, 255, 255, 0.03);
        border-left: 3px solid var(--text-muted);
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 4px;
      }

      .thought-summary {
        font-size: 0.8125rem;
        color: var(--text-muted);
        cursor: pointer;
        font-weight: 600;
        outline: none;
      }

      .thought-content {
        margin-top: 6px;
        font-size: 0.875rem;
        color: #94a3b8;
        white-space: pre-wrap;
        font-style: italic;
      }

      /* Actions bar */
      .message-actions {
        display: flex;
        gap: 8px;
        margin-top: 8px;
        opacity: 0;
        transition: opacity 0.2s;
        justify-content: flex-end;
        align-items: center;
      }

      .message-bubble:hover .message-actions {
        opacity: 1;
      }

      .btn-action {
        background: none;
        border: none;
        color: var(--text-muted);
        font-size: 0.75rem;
        cursor: pointer;
        padding: 2px 6px;
        border-radius: 4px;
        transition: all 0.15s;
        font-weight: 600;
      }

      .btn-action:hover {
        color: var(--text);
        background-color: rgba(255, 255, 255, 0.05);
      }

      .message-stats {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-right: auto;
        display: flex;
        gap: 12px;
      }
    `,
  ];

  private handleCopyMessage(e: Event) {
    const targetBtn = e.target as HTMLButtonElement;
    navigator.clipboard.writeText(this.message.text)
      .then(() => {
        targetBtn.innerText = 'Copied!';
        setTimeout(() => { targetBtn.innerText = 'Copy'; }, 2000);
      })
      .catch(err => console.error('[LiteRT-LM] Failed to copy message:', err));
  }

  private async handleRewindEdit() {
    const originalPrompt =
        await this.state.chatSession.rewindAndEdit(this.index);
    if (originalPrompt) {
      this.dispatchEvent(new CustomEvent('edit-prompt', {
        detail: {prompt: originalPrompt},
        bubbles: true,
        composed: true,
      }));
    }
  }

  override render() {
    const msg = this.message;
    const isUser = msg.role === 'user';

    // Clean size-free model names for bubble headers
    let senderName = msg.senderName;
    if (!isUser) {
      senderName = senderName
        .replace(/,\s*\d+(\.\d+)?\s*GB\)$/, ')')
        .replace(/\s*\(\d+(\.\d+)?\s*GB\)$/, '');
    }

    return html`
      <div class="message-bubble ${msg.role}" data-raw-text="${msg.text}">
        <span class="message-sender ${msg.role}">${
        isUser ? 'User' : senderName}</span>
        
        <div class="message-content">
          <!-- Thought block rendering -->
          ${
        msg.thoughtText ? html`
            <details class="thought-details" open>
              <summary class="thought-summary">Thought Process</summary>
              <div class="thought-content">${msg.thoughtText}</div>
            </details>
          ` :
                          ''}
          
          <!-- Main message bubble content -->
          ${
        isUser ? html`<div class="message-user-text">${msg.text}</div>` :
                 html`<div class="message-text">${msg.text}</div>`}
        </div>

        <!-- Message actions bar (Edit / Copy & Retry) -->
        <div class="message-actions">
          ${
        isUser ? html`
            <!-- User stats telemetry (Token count only!) -->
            ${
                 msg.tokensCount ? html`
              <div class="message-stats">
                <span>Tokens: <b>${msg.tokensCount}</b></span>
              </div>
            ` :
                                   ''}
            <!-- User Edit trigger -->
            <button class="btn-action" style="width: 58px;" @click=${
                 this.handleRewindEdit}>✎ Edit</button>
          ` :
                 html`
            <!-- Assistant stats telemetry (prefilled auto-left!) -->
            ${
                 msg.prefillSpeed || msg.decodeSpeed ? html`
              <div class="message-stats">
                <span>Prefill: <b>${msg.prefillSpeed || '-'}</b></span>
                <span>Decode: <b>${msg.decodeSpeed || '-'}</b></span>
                <span>Tokens: <b>${msg.tokensCount || '-'}</b></span>
              </div>
            ` :
                                                           ''}
            
            <!-- Copy & Retry triggers -->
            <button class="btn-action" style="width: 58px;" @click=${
                 this.handleCopyMessage}>Copy</button>
            <button class="btn-action" style="width: 58px;" @click=${
                 () => this.state.chatSession.redoResponse(
                     this.index)}>⟲ Retry</button>
          `}
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-chat-bubble': LitertChatBubble;
  }
}
