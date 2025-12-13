# S3 Bucket Variables
variable "results_bucket_name" {
    description = "Name of the S3 bucket for results"
    type        = string
    default     = "results-bucket"
}

variable "preprocessing_bucket_name" {
    description = "Name of the S3 bucket for preprocessing"
    type        = string
    default     = "preprocessing-bucket"
}

variable "versioning_status" {
    description = "Versioning status for the S3 buckets"
    type        = string
    default     = "Enabled"
}

variable "sse_algorithm" {
    description = "Server-side encryption algorithm for the S3 buckets"
    type        = string
    default     = "AES256"
}