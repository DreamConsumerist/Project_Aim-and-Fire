## Project Aim and Fire
Unity 3D FPS project that uses the classic, intuitive finger gun as the control scheme. Pointing at the screen allows you to aim, curling your fingers allows you to fire, 
and whatever other gestures I decide to add could do all manner of things as well.

## Project Objectives
1. To be my semester project for Computer Vision at UAA, taught by Dr. Masoumeh Heidari Kapourchali.
2. To work in Unity on an interesting new way to play.

## Pipeline
In order to use MediaPipe Hands, you have to have a camera. For MediaPipe Hands to work as well as possible, you have to have a fairly decent camera. Once that step is cleared, we can talk about how we can use it.
1. On booting up the game, a Python script will begin running that opens a stream of video input to MediaPipe Hands, which will generate vectors containing landmark locations for joints in your hands.
2. A TCN model will consume the vectors and check what state your hand is currently in, aiming, firing, etc, classifying it for gameplay purposes.
3. This Python script sends UDP packets containing the MediaPipe Hands landmark data as well as the current gesture classification over to the Unity game which has a socket open to receive them.
4. Unity parses the data into game variables to calculate aim trajectory and control the current game state.

## Current Status
Developing in a mad dash to submit on time in 3 days... I will update with gameplay if I get it done in time.
