output "preprocessing_bucket_name" {
    value = aws_s3_bucket.preprocessing.bucket
}

output "results_bucket_name" {
    value = aws_s3_bucket.results.bucket
}