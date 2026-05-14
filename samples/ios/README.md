# LiteRT-LM Swift API Usage for iOS

This directory contains examples and instructions on how to use the LiteRT-LM
Swift API to integrate on-device Large Language Models into your iOS
applications.

## Prerequisites

- iOS 15.0 or later
- Xcode 15.0 or later
- A `.litertlm` model file (e.g., Gemma)

## Integration Steps

### 1. Add Dependency

Add the LiteRT-LM package to your Xcode project using Swift Package Manager:
1. In Xcode, select **File** > **Add Package Dependencies...**
2. In the search bar at the top right, enter the GitHub repository URL:
   `https://github.com/google-ai-edge/LiteRT-LM.git`
3. Select the package from the list and click **Add Package**.
4. Select the target app you want to add the dependency to and click **Finish**.

### 2. Add a Model File

1. Obtain a compatible `.litertlm` model file.
2. Drag and drop the model file into your Xcode project navigator. In the
   dialog that appears, ensure your app target is checked.
3. *(Alternative)* If the file is not found at runtime, go to your project's
   **Build Phases** > **Copy Bundle Resources** and add the file there
   manually.

### 3. Usage Example

Simple example of a SwiftUI chat application using streaming responses is
available in the [ContentView.swift](ContentView.swift) file in this
directory.

It demonstrates how to:

- Find the model in the app bundle.
- Configure and initialize the `Engine`.
- Create a `Conversation` session.
- Send a message and handle the streaming response live in the UI.

## GPU Acceleration (Optional)

To get the best performance on supported iOS devices using the **dynamically
linked C API**, you can enable GPU acceleration:

1. Ensure your `EngineConfig` uses `backend: .gpu`.
2. You must also include the following dynamic libraries in your app bundle for
   full GPU support:
   - `libLiteRtTopKMetalSampler.dylib` (if your model requires TopK sampling on GPU).
   - `libLiteRtMetalAccelerator.dylib` (for Metal GPU acceleration).
   **Steps to add the libraries**:
   - Copy the `.dylib` files to your local **Desktop**.
   - In Xcode, go to your project's **Build Phases** tab.
   - Click the **`+`** button at the top left of the Build Phases area and
     select **New Copy Files Phase**.
   - Set the **Destination** to **Frameworks** and leave the subpath empty.
   - Click the **`+`** button at the bottom of the new Copy Files section.
   - Click **Add Other...** and navigate to your **Desktop**.
   - Select `libLiteRtTopKMetalSampler.dylib` and add it.
   - Click the **`+`** button again to add `libLiteRtMetalAccelerator.dylib`
     in the same way.

## Running the App

1. Connect your physical iPhone to your Mac using a cable.
2. In Xcode, select your iPhone's name from the run destination menu at the
   top center of the window.
3. Click the **Run** button (or press `Cmd + R`) to build and run the app on
   your device!
