using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

namespace SpellDrawing
{
    /// <summary>Captures mouse drags as strokes, rendering each with its own trail. Strokes accumulate
    /// into one gesture until the cast key fires OnGestureCompleted; the cancel key discards them.</summary>
    [RequireComponent(typeof(LineRenderer))]
    public class StrokeDrawer : MonoBehaviour
    {
        [Header("Capture")]
        [Tooltip("Minimum distance in screen pixels between recorded points. Higher = fewer points, coarser gesture.")]
        [SerializeField] private float minPointDistance = 6f;

        [Tooltip("Minimum number of points a single stroke needs to be kept — filters out accidental clicks.")]
        [SerializeField] private int minPointsToRecognize = 6;

        [Header("Multi-Stroke")]
        [Tooltip("Finalizes every stroke drawn so far into one gesture and fires OnGestureCompleted.")]
        [SerializeField] private Key castKey = Key.Space;

        [Tooltip("Discards every stroke drawn so far without casting.")]
        [SerializeField] private Key cancelKey = Key.Escape;

        [Header("Visuals")]
        [SerializeField] private Color strokeColor = new Color(0.4f, 0.8f, 1f, 0.9f);
        [SerializeField] private float strokeWidth = 0.08f;
        [Tooltip("Camera used to convert mouse screen position to world space for the trail. Defaults to Camera.main.")]
        [SerializeField] private Camera drawCamera;

        private Material _lineMaterial;
        private LineRenderer _activeLine;
        private readonly List<LineRenderer> _completedLines = new List<LineRenderer>();
        private readonly List<Vector2> _currentStrokePoints = new List<Vector2>();
        private readonly List<List<Vector2>> _completedStrokes = new List<List<Vector2>>();
        private bool _isDrawing;

        public event Action OnStrokeStarted;

        /// <summary>Fires on cast with every stroke, kept separate. Flatten() for matching; keep
        /// separate for rendering/rasterizing.</summary>
        public event Action<List<List<Vector2>>> OnGestureCompleted;

        public static List<Vector2> Flatten(IReadOnlyList<List<Vector2>> strokes)
        {
            var combined = new List<Vector2>();
            foreach (var stroke in strokes) combined.AddRange(stroke);
            return combined;
        }

        public bool IsDrawing => _isDrawing;
        public IReadOnlyList<Vector2> CurrentStrokeScreenPoints => _currentStrokePoints;
        public int PendingStrokeCount => _completedStrokes.Count;

        private void Awake()
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Sprites/Default");
            _lineMaterial = new Material(shader);

            _activeLine = GetComponent<LineRenderer>();
            ConfigureLine(_activeLine);

            if (drawCamera == null) drawCamera = Camera.main;
        }

        private void OnDestroy()
        {
            if (_lineMaterial != null) Destroy(_lineMaterial);
        }

        private void Update()
        {
            HandleMouse();
            HandleKeys();
        }

        private void HandleMouse()
        {
            Mouse mouse = Mouse.current;
            if (mouse == null) return;

            if (mouse.leftButton.wasPressedThisFrame)
            {
                BeginStroke(mouse.position.ReadValue());
            }
            else if (_isDrawing && mouse.leftButton.isPressed)
            {
                ContinueStroke(mouse.position.ReadValue());
            }
            else if (_isDrawing && mouse.leftButton.wasReleasedThisFrame)
            {
                EndStroke();
            }
        }

        private void HandleKeys()
        {
            Keyboard keyboard = Keyboard.current;
            if (keyboard == null) return;

            if (keyboard[castKey].wasPressedThisFrame) TryCompleteGesture();
            else if (keyboard[cancelKey].wasPressedThisFrame) ClearGesture();
        }

        private void BeginStroke(Vector2 screenPos)
        {
            _isDrawing = true;
            _currentStrokePoints.Clear();
            _currentStrokePoints.Add(screenPos);

            _activeLine.positionCount = 0;
            AppendLinePoint(_activeLine, screenPos);

            OnStrokeStarted?.Invoke();
        }

        private void ContinueStroke(Vector2 screenPos)
        {
            if (Vector2.Distance(_currentStrokePoints[_currentStrokePoints.Count - 1], screenPos) < minPointDistance) return;

            _currentStrokePoints.Add(screenPos);
            AppendLinePoint(_activeLine, screenPos);
        }

        private void EndStroke()
        {
            _isDrawing = false;

            if (_currentStrokePoints.Count < minPointsToRecognize)
            {
                _activeLine.positionCount = 0;
                return;
            }

            LineRenderer baked = CreateSegmentLine();
            baked.positionCount = _activeLine.positionCount;
            for (int i = 0; i < _activeLine.positionCount; i++) baked.SetPosition(i, _activeLine.GetPosition(i));
            _completedLines.Add(baked);
            _activeLine.positionCount = 0;

            _completedStrokes.Add(new List<Vector2>(_currentStrokePoints));
        }

        private void TryCompleteGesture()
        {
            if (_completedStrokes.Count == 0) return;

            var strokes = new List<List<Vector2>>(_completedStrokes);

            ClearGesture();
            OnGestureCompleted?.Invoke(strokes);
        }

        public void ClearGesture()
        {
            _completedStrokes.Clear();
            foreach (var line in _completedLines) Destroy(line.gameObject);
            _completedLines.Clear();
            _activeLine.positionCount = 0;
        }

        private LineRenderer CreateSegmentLine()
        {
            var go = new GameObject("StrokeSegment");
            go.transform.SetParent(transform, false);
            var line = go.AddComponent<LineRenderer>();
            ConfigureLine(line);
            return line;
        }

        private void ConfigureLine(LineRenderer line)
        {
            line.positionCount = 0;
            line.useWorldSpace = true;
            line.startWidth = strokeWidth;
            line.endWidth = strokeWidth;
            line.sharedMaterial = _lineMaterial;
            line.startColor = strokeColor;
            line.endColor = strokeColor;
        }

        private void AppendLinePoint(LineRenderer line, Vector2 screenPos)
        {
            line.positionCount++;
            line.SetPosition(line.positionCount - 1, ScreenToDrawPlane(screenPos));
        }

        private Vector3 ScreenToDrawPlane(Vector2 screenPos)
        {
            if (drawCamera == null) return new Vector3(screenPos.x, screenPos.y, 0f);

            float distance = Mathf.Abs(drawCamera.transform.position.z);
            Vector3 world = drawCamera.ScreenToWorldPoint(new Vector3(screenPos.x, screenPos.y, distance));
            world.z = 0f;
            return world;
        }
    }
}
