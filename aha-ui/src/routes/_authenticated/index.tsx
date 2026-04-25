import { createFileRoute } from '@tanstack/react-router'
import { ModelsPage } from '@/features/models'

export const Route = createFileRoute('/_authenticated/')({
  component: ModelsPage,
})
