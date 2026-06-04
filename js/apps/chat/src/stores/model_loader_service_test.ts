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

import {ModelLoaderService} from './model_loader_service.js';
import {SettingsStore} from './settings_store.js';

describe('ModelLoaderService', () => {
  let settingsStore: SettingsStore;
  let modelLoader: ModelLoaderService;

  beforeEach(() => {
    settingsStore = new SettingsStore(() => {});
    modelLoader =
        new ModelLoaderService(() => {}, settingsStore, (msg: string) => {});
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it('initializes neatly', () => {
    expect(modelLoader.isModelLoading).toBeFalse();
    expect(modelLoader.cachedModels.size).toBe(0);
  });

  it('aborts active downloads neatly', () => {
    const mockAbort = jasmine.createSpy('abort');
    const MOCK_ABORT_CONTROLLER = {abort: mockAbort} as unknown as
        AbortController;

    modelLoader.downloadAbortController = MOCK_ABORT_CONTROLLER;
    modelLoader.cancelDownload();

    expect(modelLoader.isDownloadAborted).toBeTrue();
    expect(mockAbort).toHaveBeenCalled();
  });
});
