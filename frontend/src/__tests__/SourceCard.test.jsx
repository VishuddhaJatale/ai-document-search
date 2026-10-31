import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import SourceCard from '../components/SourceCard'

describe('SourceCard', () => {
  it('groups pages by filename and removes duplicate page labels', () => {
    render(<SourceCard sources={[{ filename: 'policy.pdf', page: 1 }, { filename: 'policy.pdf', page: 2 }, { filename: 'policy.pdf', page: 1 }, { filename: 'fees.txt', page: 4 }]} />)
    expect(screen.getByText('policy.pdf')).toBeInTheDocument()
    expect(screen.getByText('fees.txt')).toBeInTheDocument()
    expect(screen.getByText(/Pages 1 · 2/)).toBeInTheDocument()
  })
})
