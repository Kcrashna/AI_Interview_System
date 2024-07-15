#filters the questions based on job description
def filter_questions(df, job_description):
    if job_description.lower() == 'backend':
        return df[df['Category'].str.lower() == 'backend']
    elif job_description.lower() == 'frontend':
        return df[df['Category'].str.lower() == 'frontend']
    elif job_description.lower() == 'mern stack':
        return df  # Return all questions if job description is MERN stack
    else:
        raise ValueError("Unknown job description")
