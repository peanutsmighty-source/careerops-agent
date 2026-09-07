from __future__ import annotations

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from app.models import JobRequirement, Skill


def skill_demand_rows(session: Session) -> list[dict]:
    rows = session.execute(
        select(
            Skill.id.label("skill_id"),
            Skill.name,
            Skill.category,
            func.count(JobRequirement.id).label("requirement_count"),
            func.count(distinct(JobRequirement.job_id)).label("job_count"),
        )
        .join(JobRequirement, JobRequirement.skill_id == Skill.id)
        .group_by(Skill.id, Skill.name, Skill.category)
        .order_by(
            func.count(distinct(JobRequirement.job_id)).desc(),
            func.count(JobRequirement.id).desc(),
            Skill.name,
        )
    )
    return [
        {
            "skill_id": row.skill_id,
            "name": row.name,
            "category": row.category,
            "requirement_count": row.requirement_count,
            "job_count": row.job_count,
        }
        for row in rows
    ]
