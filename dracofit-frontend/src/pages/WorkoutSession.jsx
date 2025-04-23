import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import workoutsService from "../services/workoutsService";
import workoutExercisesService from "../services/workoutExercisesService"; // Import the new service
import workoutLogsService from "../services/workoutLogsService";
import exerciseLogsService from "../services/exerciseLogsService";
import Timer from "../components/common/Timer";
import CurrentExercisePanel from "../components/workouts/CurrentExercisePanel";

const WorkoutSession = () => {
  const { workoutId } = useParams();
  const navigate = useNavigate();
  const { currentUser } = useAuth();

  const [loading, setLoading] = useState(true);
  const [workout, setWorkout] = useState(null);
  const [workoutLog, setWorkoutLog] = useState(null);
  const [currentExerciseIndex, setCurrentExerciseIndex] = useState(0);
  const [completedExercises, setCompletedExercises] = useState([]);
  const [startTime, setStartTime] = useState(null);
  const [error, setError] = useState(null);
  const [currentSetIndex, setCurrentSetIndex] = useState(0);
  const [currentExerciseSetData, setCurrentExerciseSetData] = useState([]); // Store sets for the current exercise
  const [notification, setNotification] = useState({
    show: false,
    message: "",
    type: "",
  });

  const showNotification = (message, type = "success") => {
    setNotification({ show: true, message, type });
    setTimeout(() => {
      setNotification({ show: false, message: "", type: "" });
    }, 3000);
  };

  // Fetch workout details and create initial workout log
  useEffect(() => {
    const fetchWorkoutDetails = async () => {
      try {
        setLoading(true);

        // Step 1: Get workout plan data
        console.log(`Fetching workout plan with ID: ${workoutId}`);
        const workoutData = await workoutsService.getWorkoutById(workoutId);
        console.log("Workout plan data received:", workoutData);

        // Step 2: Get workout exercises separately
        console.log(`Fetching exercises for workout plan ${workoutId}`);
        const exercisesData = await workoutExercisesService.getWorkoutExercises(
          workoutId
        );
        console.log("Workout exercises received:", exercisesData);

        // Step 3: Combine the data
        const combinedData = {
          ...workoutData,
          exercises: exercisesData.map((item) => {
            // Extract exercise details and workout-exercise relationship details
            const exercise = item.exercise || {};
            return {
              id: exercise.id, // Exercise ID
              exerciseId: exercise.id, // Duplicate for flexibility in access
              name: exercise.name || "Unknown Exercise",
              muscleGroup: exercise.muscleGroup || exercise.target || "Various",
              instructions: exercise.instructions || "",
              sets: item.sets || 3, // Sets from workout-exercise relationship
              reps: item.reps || 10, // Reps from workout-exercise relationship
              restTimeSeconds: item.restTimeSeconds || 60,
            };
          }),
        };

        console.log("Combined workout data with exercises:", combinedData);

        // Verify we have exercises
        if (!combinedData.exercises || combinedData.exercises.length === 0) {
          console.error("No exercises found in the workout data");
          setError(
            "This workout has no exercises. Please add exercises to the workout first."
          );
          setLoading(false);
          return;
        }

        setWorkout(combinedData);

        // Create the initial workout log with 0 duration
        console.log(
          "Creating initial workout log with workoutPlanId:",
          parseInt(workoutId)
        );
        const initialWorkoutLog = await workoutLogsService.startWorkout({
          workoutPlanId: parseInt(workoutId),
          durationMinutes: 0, // Start with 0 duration (using minutes now)
        });

        console.log("Initial workout log created:", initialWorkoutLog);
        setWorkoutLog(initialWorkoutLog);
        setStartTime(new Date());
        setLoading(false);
      } catch (err) {
        console.error("Error initializing workout:", err);
        if (err.response) {
          console.error("Error response status:", err.response.status);
          console.error("Error response data:", err.response.data);
        }
        setError(`Failed to start workout: ${err.message || "Unknown error"}`);
        setLoading(false);
      }
    };

    fetchWorkoutDetails();
  }, [workoutId]);

  // Handle exercise set completion
  const handleSetComplete = async (setData) => {
    // setData likely contains { reps, weight } from CurrentExercisePanel
    try {
      const exercises = workout?.exercises || [];
      const currentExercise = exercises[currentExerciseIndex];
      const totalSets = currentExercise?.sets || 3; // Default to 3 if not specified

      if (!currentExercise || !workoutLog?.id) {
        console.error("Missing current exercise or workout log ID");
        showNotification("Error: Cannot record set.", "error");
        return;
      }

      // --- Accumulate Set Data Locally ---
      const newSetEntry = {
        setNumber: currentSetIndex + 1, // Set numbers are usually 1-based
        reps: parseInt(setData.reps) || 0, // Ensure reps is a number
        weight:
          setData.weight !== undefined && setData.weight !== ""
            ? parseInt(setData.weight)
            : undefined, // Ensure weight is a number or undefined
      };
      const updatedSetData = [...currentExerciseSetData, newSetEntry];
      setCurrentExerciseSetData(updatedSetData);
      console.log(
        `Accumulated set data for exercise ${currentExercise.id}:`,
        updatedSetData
      );
      // --- End Accumulation ---

      // Check if this was the last set for the current exercise
      if (currentSetIndex >= totalSets - 1) {
        // --- Last Set Completed: Create ExerciseLog with all sets ---
        console.log(
          `Last set for exercise ${currentExercise.id} completed. Saving ExerciseLog.`
        );
        try {
          const payload = {
            exerciseId: currentExercise.id,
            setsData: updatedSetData, // Send the accumulated sets
          };
          console.log("Sending payload to createExerciseLog:", payload);

          // Assuming createExerciseLog takes workoutLogId and the payload
          const createdExerciseLog =
            await exerciseLogsService.createExerciseLog(workoutLog.id, payload);

          console.log("ExerciseLog created successfully:", createdExerciseLog);
          showNotification(
            `Exercise ${currentExercise.name} completed and logged!`
          );

          // Reset set data for the next exercise
          setCurrentExerciseSetData([]);
          setCurrentSetIndex(0); // Reset set index

          // Move to the next exercise or finish workout
          if (currentExerciseIndex < exercises.length - 1) {
            setCompletedExercises([
              ...completedExercises,
              currentExerciseIndex,
            ]);
            setCurrentExerciseIndex(currentExerciseIndex + 1);
          } else {
            // Last exercise of the workout
            setCompletedExercises([
              ...completedExercises,
              currentExerciseIndex,
            ]);
            handleCompleteWorkout(); // Trigger workout completion
          }
        } catch (createErr) {
          console.error("Error creating exercise log:", createErr);
          if (createErr.response) {
            console.error("Server Response:", createErr.response.data);
            showNotification(
              `Failed to save exercise log: ${
                createErr.response.data?.message?.join(", ") ||
                createErr.message
              }`,
              "error"
            );
          } else {
            showNotification(
              `Failed to save exercise log: ${createErr.message}`,
              "error"
            );
          }
          // Optional: Decide if you want to rollback state or allow user to retry?
          // For now, we'll just show the error. The accumulated data is still in state.
          // To prevent moving on, you might return here:
          // return;
        }
        // --- End ExerciseLog Creation ---
      } else {
        // --- Not the last set: Move to the next set ---
        setCurrentSetIndex(currentSetIndex + 1);
        showNotification(`Set ${currentSetIndex + 1} completed! Next set.`);
      }
    } catch (err) {
      console.error("Error in handleSetComplete:", err);
      showNotification(`An unexpected error occurred: ${err.message}`, "error");
    }
  };

  // Complete the workout - update the existing log with the final duration
  const handleCompleteWorkout = async () => {
    try {
      if (!workoutLog || !workoutLog.id) {
        console.error("No workout log found to update");
        showNotification("Error: No active workout found", "error");
        return;
      }

      const endTime = new Date();
      // Calculate duration in minutes instead of seconds
      const durationSeconds = Math.round((endTime - startTime) / 1000);
      const durationMinutes = Math.round(durationSeconds / 60);

      console.log(
        "Completing workout, duration in seconds:",
        durationSeconds,
        "converted to minutes:",
        durationMinutes
      );

      // Update the existing workout log with the final duration and completed status
      const completedWorkout = await workoutLogsService.completeWorkout(
        workoutLog.id,
        {
          durationMinutes: durationMinutes,
        }
      );

      console.log("Workout completed successfully:", completedWorkout);
      showNotification("Workout completed successfully!");

      // Navigate to the workout summary
      setTimeout(() => {
        navigate(`/workout-summary/${workoutLog.id}`);
      }, 1500);
    } catch (err) {
      console.error("Error completing workout:", err);
      showNotification(`Failed to complete workout: ${err.message}`, "error");
    }
  };

  // Abandon the workout
  const handleAbandonWorkout = async () => {
    if (window.confirm("Are you sure you want to abandon this workout?")) {
      try {
        if (workoutLog && workoutLog.id) {
          await workoutLogsService.abandonWorkout(workoutLog.id);
        }
        showNotification("Workout abandoned.");
        setTimeout(() => {
          navigate("/workouts");
        }, 1000);
      } catch (err) {
        console.error("Error abandoning workout:", err);
        showNotification("Failed to abandon workout.", "error");
      }
    }
  };

  if (loading) {
    return (
      <div className="p-4 text-center text-goldenrod">
        Loading workout session...
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-center text-red-500">
        <p>{error}</p>
        <button
          onClick={() => navigate("/workouts")}
          className="mt-4 bg-goldenrod hover:bg-dark-goldenrod text-midnight-green py-2 px-4 rounded-md"
        >
          Back to Workouts
        </button>
      </div>
    );
  }

  const exercises = workout?.exercises || [];
  const currentExercise = exercises[currentExerciseIndex];
  const totalSets = currentExercise?.sets || 3;

  return (
    <div className="p-4 max-w-lg mx-auto bg-midnight-green min-h-screen">
      {/* Notification */}
      {notification.show && (
        <div
          className={`fixed top-4 right-4 p-3 rounded-md shadow-md z-50 ${
            notification.type === "error" ? "bg-red-600" : "bg-dark-slate-gray"
          } text-goldenrod`}
        >
          {notification.message}
        </div>
      )}

      <h1 className="text-2xl font-bold mb-4 text-goldenrod">
        {workout?.name}
      </h1>

      {/* Timer */}
      <div className="mb-4">
        <Timer startTime={startTime} />
      </div>

      {/* Overall Progress indicator */}
      <div className="mb-6">
        <div className="flex justify-between text-sm text-gray mb-1">
          <span>
            Exercise {currentExerciseIndex + 1} of {exercises.length}
          </span>
          <span>
            {Math.round((completedExercises.length / exercises.length) * 100)}%
            complete
          </span>
        </div>
        <div className="w-full bg-midnight-green-darker rounded-full h-2.5">
          <div
            className="bg-medium-aquamarine h-2.5 rounded-full"
            style={{
              width: `${(completedExercises.length / exercises.length) * 100}%`,
            }}
          ></div>
        </div>
      </div>

      {/* Current exercise panel */}
      {currentExercise && (
        <CurrentExercisePanel
          exercise={currentExercise}
          currentSetIndex={currentSetIndex}
          totalSets={totalSets}
          onCompleteSet={handleSetComplete}
          isLastExercise={currentExerciseIndex === exercises.length - 1}
          isLastSet={currentSetIndex === totalSets - 1}
        />
      )}

      {/* Abandon workout button */}
      <button
        onClick={handleAbandonWorkout}
        className="w-full bg-sepia hover:bg-dark-goldenrod text-goldenrod py-2 rounded-md mt-6"
      >
        Abandon Workout
      </button>

      {/* Debug Information (only in development) */}
      {process.env.NODE_ENV !== "production" && (
        <div className="mt-8 p-4 border border-dashed border-gray-600 rounded-lg">
          <h3 className="text-sm font-semibold text-gray-400 mb-2">
            Debug Info
          </h3>
          <div className="text-xs text-gray-500">
            <p>Workout ID: {workoutId}</p>
            <p>Workout Log ID: {workoutLog?.id || "Not created"}</p>
            <p>Exercise Count: {exercises?.length || 0}</p>
            <p>Current Exercise: {currentExerciseIndex + 1}</p>
            {startTime && (
              <p>
                Current Duration: {Math.round((new Date() - startTime) / 60000)}{" "}
                minutes
              </p>
            )}
            <details className="mt-2">
              <summary className="cursor-pointer text-gray-400">
                Exercise Data
              </summary>
              <pre className="mt-2 overflow-auto max-h-40 text-xs p-2 bg-gray-900 rounded">
                {JSON.stringify(exercises, null, 2)}
              </pre>
            </details>
          </div>
        </div>
      )}
    </div>
  );
};

export default WorkoutSession;
