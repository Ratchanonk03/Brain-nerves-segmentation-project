resource "aws_s3_bucket" "preprocessing" {
    bucket = var.preprocessing_bucket_name
    force_destroy = true
}

resource "aws_s3_bucket_versioning" "preprocessing_bucket_versioning" {
    bucket = aws_s3_bucket.preprocessing.id
    versioning_configuration {
        status = var.versioning_status
    }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "preprocessing_bucket_encryption_configuration" {
    bucket = aws_s3_bucket.preprocessing.id
    rule {
        apply_server_side_encryption_by_default {
            sse_algorithm = var.sse_algorithm
        }
    }
}

resource "aws_s3_bucket" "results" {
    bucket = var.results_bucket_name
    force_destroy = true
}

resource "aws_s3_bucket_versioning" "results_bucket_versioning" {
    bucket = aws_s3_bucket.results.id
    versioning_configuration {
        status = var.versioning_status
    }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "results_bucket_encryption_configuration" {
    bucket = aws_s3_bucket.results.id
    rule {
        apply_server_side_encryption_by_default {
            sse_algorithm = var.sse_algorithm
        }
    }
}