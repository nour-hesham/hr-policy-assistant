# Sample HR Policy Q&A

These six examples preserve unchanged outputs from the final retrieval and
generation pipeline. They cover direct, conditional, multi-section,
cross-policy, partial, and unsupported questions.

## 1. Conditional numeric rule

**Question**

How many days of annual leave am I entitled to after 6 years of service?

**Status:** `answered`

**Produced answer**

After 6 years of continuous service, you are entitled to 25 working days of paid annual leave per calendar year [S1].

**Sources**

- [S1] Annual Leave Policy - Entitlement (HR-POL-001, version 2025-03, `01-annual-leave.md`)

## 2. Direct procedural answer

**Question**

What is the procedure for filing a formal grievance?

**Status:** `answered`

**Produced answer**

To file a formal grievance, an employee must submit a formal written complaint to HR within 30 calendar days of the event or the informal resolution step, whichever is later [S1]. The complaint must include a clear description of the concern, relevant dates, people involved, supporting information, and the desired outcome [S1]. HR will acknowledge receipt of the complaint within 3 business days [S1].

**Sources**

- [S1] Grievance and Complaints Procedure - Step 2: Formal written complaint (HR-POL-008, version 2025-02, `08-grievance-procedure.md`)

## 3. Multiple sections from one policy

**Question**

What happens if I work on a weekend — do I get extra pay?

**Status:** `answered`

**Produced answer**

If you work on a weekend, you may be eligible for overtime pay if you are in a non-exempt role and the work is pre-approved [S2]. For non-exempt roles, weekend overtime is paid at 2.0 times the standard hourly rate, though time in lieu is preferred over cash payment whenever practical [S2]. Exempt roles are not eligible for overtime payment, though time-in-lieu arrangements may be available at a manager's discretion in exceptional circumstances [S2]. All weekend work must be pre-approved by both your line manager and department head [S1].

**Sources**

- [S1] Work Hours and Overtime Policy - Weekend and public-holiday work (HR-POL-004, version 2025-01, `04-work-hours-overtime.md`)
- [S2] Work Hours and Overtime Policy - Overtime (HR-POL-004, version 2025-01, `04-work-hours-overtime.md`)

## 4. Cross-policy answer

**Question**

When I work remotely, how many days per week can I do it and during what core hours must I be available?

**Status:** `answered`

**Produced answer**

Eligible employees may work remotely for up to 2 days per calendar week [S1]. Regardless of whether you are working in the office or remotely, you are expected to be available during the core hours of 10:00 AM to 4:00 PM [S2].

**Sources**

- [S1] Remote Work Policy - Standard arrangement (HR-POL-003, version 2025-02, `03-remote-work.md`)
- [S2] Work Hours and Overtime Policy - Core hours (HR-POL-004, version 2025-01, `04-work-hours-overtime.md`)

## 5. Partial answer

**Question**

What is the annual training budget, and what paid parental leave does the company offer?

**Status:** `partial`

**Produced answer**

Each permanent employee has an annual training budget of USD 2,500 per calendar year [S1]. The provided policy documents do not contain information regarding paid parental leave.

**Sources**

- [S1] Learning and Development Policy - Annual training budget (HR-POL-009, version 2025-03, `09-learning-development.md`)

## 6. Unsupported question

**Question**

What is the capital of Japan?

**Status:** `not_found`

**Produced answer**

I couldn't find this information in the supplied HR policies.

**Sources**

None - the system abstained without exposing evidence.
