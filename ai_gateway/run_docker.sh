docker run \
  --mount type=bind,source=$(pwd)/app/result,target=/app/result \
  --env-file /Users/ratchanonkhongsawi/Desktop/CMKL/3rd/Scalable/assessment/assessment_2/.env \
  -p 8000:8000 \
  gateway:local