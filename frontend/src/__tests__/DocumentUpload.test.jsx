import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import DocumentUpload from '../components/DocumentUpload'

vi.mock('../services/api', () => ({ uploadDocument: vi.fn() }))

describe('DocumentUpload validation', () => {
  it('rejects unsupported file types', () => {
    render(<DocumentUpload onUploadSuccess={vi.fn()} />)
    fireEvent.change(document.querySelector('input[type=file]'), { target: { files: [new File(['x'], 'notes.doc')] } })
    expect(screen.getByText(/Only PDF and TXT/i)).toBeInTheDocument()
  })
  it('rejects files larger than 10 MB', () => {
    render(<DocumentUpload onUploadSuccess={vi.fn()} />)
    const file = new File([new Uint8Array(10 * 1024 * 1024 + 1)], 'large.txt')
    fireEvent.change(document.querySelector('input[type=file]'), { target: { files: [file] } })
    expect(screen.getByText(/10 MB or smaller/i)).toBeInTheDocument()
  })
  it('rejects empty files', () => {
    render(<DocumentUpload onUploadSuccess={vi.fn()} />)
    fireEvent.change(document.querySelector('input[type=file]'), { target: { files: [new File([], 'empty.txt')] } })
    expect(screen.getByText(/selected file is empty/i)).toBeInTheDocument()
  })
})
